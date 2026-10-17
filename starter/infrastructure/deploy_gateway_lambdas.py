import io, os, zipfile, boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
P = os.environ.get("PROJECT_NAME", "udacity-agentcore")
iam, lam = boto3.client("iam"), boto3.client("lambda", region_name=REGION)

HANDLERS = {
"orders-api": '''
import boto3, os
from boto3.dynamodb.conditions import Attr
def handler(event, context):
    t = boto3.resource("dynamodb").Table(os.environ["ORDERS_TABLE"])
    items = t.scan(FilterExpression=Attr("order_id").eq(event["order_id"].upper())).get("Items", [])
    return {"status": "ok", "order": items[0] if items else None}
''',
"customers-api": '''
import boto3, os
def handler(event, context):
    t = boto3.resource("dynamodb").Table(os.environ["CUSTOMERS_TABLE"])
    c = t.get_item(Key={"customer_id": event["customer_id"]}).get("Item")
    return {"status": "ok", "customer": c and {"customer_id": c["customer_id"], "tier": c["tier"], "name": c["name"]}}
''',
"policy-api": '''
import boto3, os
def handler(event, context):
    rt = boto3.client("bedrock-agent-runtime")
    out = []
    for kb in ("RETURNS_KB_ID", "SHIPPING_KB_ID", "WARRANTY_KB_ID"):
        if os.environ.get(kb):
            r = rt.retrieve(knowledgeBaseId=os.environ[kb], retrievalQuery={"text": event["query"]},
                            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 3}})
            out += [x["content"]["text"] for x in r["retrievalResults"]]
    return {"status": "ok", "passages": out}
''',
}

role_arn = os.environ.get("LAMBDA_ROLE_ARN")
if not role_arn:   # needs iam:CreateRole; if your lab blocks it, set LAMBDA_ROLE_ARN to an existing Lambda role
    name = f"{P}-gateway-lambda-role"
    role_arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=(
        '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    ))["Role"]["Arn"]
    iam.attach_role_policy(RoleName=name, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    iam.put_role_policy(RoleName=name, PolicyName="data", PolicyDocument=(
        '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["dynamodb:Scan","dynamodb:GetItem","bedrock:Retrieve"],"Resource":"*"}]}'))
    import time; time.sleep(10)   # IAM propagation

env = {"ORDERS_TABLE": f"{P}-orders", "CUSTOMERS_TABLE": f"{P}-customers",
       **{k: os.environ.get(k, "") for k in ("RETURNS_KB_ID", "SHIPPING_KB_ID", "WARRANTY_KB_ID")}}
env = {k: v for k, v in env.items() if v}

for fn, code in HANDLERS.items():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z: z.writestr("index.py", code)
    try:
        lam.create_function(FunctionName=f"{P}-{fn}", Runtime="python3.12", Role=role_arn,
            Handler="index.handler", Code={"ZipFile": buf.getvalue()}, Environment={"Variables": env}, Timeout=30)
        print("created", f"{P}-{fn}")
    except lam.exceptions.ResourceConflictException:
        print("exists", f"{P}-{fn}")