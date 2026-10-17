"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude Haiku 4.5 - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.

Commands:
  python src/agent_orchestrator.py test            # 3 scenarios, local run, traced to X-Ray
  python src/agent_orchestrator.py chat            # interactive terminal chat
  python src/agent_orchestrator.py deploy          # Tasks 3-6 deployment pipeline (uses the AgentCore CLI)
  python src/agent_orchestrator.py invoke "<msg>"  # call the deployed AgentCore Runtime
  python src/agent_orchestrator.py serve           # HTTP server (what AgentCore Runtime runs)

Deployment uses the AgentCore CLI (`agentcore`, npm package @aws/agentcore,
https://github.com/aws/agentcore-cli) through the pre-written helper
src/agentcore_cli.py - see the README for the prerequisites (Node.js 20+, uv).
"""

import boto3
import json
import re
import time
import os
import sys
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)

# ─────────────────────────────────────────────────────
# OBSERVABILITY
# ─────────────────────────────────────────────────────
# `tool` is the Strands @tool decorator wrapped so that every tool call is
# recorded as an X-Ray subsegment (the orchestrator's route_to_* tools become
# the worker-agent nodes on the X-Ray Service Map) and logged at INFO level.
# Use it exactly like `strands.tool`:  @tool  above each tool function.
from agent_observability import (
    tool, tracer, setup_logging, flush_logs, print_trace_hint,
    apply_observability_config, wait_for_runtime_ready,
)


# ─────────────────────────────────────────────────────
# AWS CLIENTS
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  SMALL SHARED HELPERS (used by the Task 2 agents)
# ═══════════════════════════════════════════════════════

def _json_safe(value):
    """Recursively convert DynamoDB Decimals (and sets) into JSON-friendly types."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in value]
    return value


def _cfg(*names, default=None):
    """Return the first config attribute that exists and is non-empty."""
    for name in names:
        val = getattr(config, name, None)
        if val:
            return val
    return default


def _orders_table():
    return dynamodb.Table(_cfg('ORDERS_TABLE', 'ORDERS_TABLE_NAME'))


def _customers_table():
    return dynamodb.Table(_cfg('CUSTOMERS_TABLE', 'CUSTOMERS_TABLE_NAME'))


def _ensure_workflow_state(session_id: str, customer_id: str = 'unknown') -> dict:
    """Return the session's WorkflowState, creating it if initialize_session was skipped."""
    state = _read_workflow_state(session_id)
    if state is None:
        try:
            _create_workflow_state(session_id, customer_id)
        except ClientError as exc:
            if exc.response['Error']['Code'] != 'ConditionalCheckFailedException':
                raise
        state = _read_workflow_state(session_id)
    return state


def _guardrail_kwargs() -> dict:
    """BedrockModel guardrail settings, applied when the model is created (empty until known)."""
    guardrail_id      = getattr(config, 'GUARDRAIL_ID', None)
    guardrail_version = getattr(config, 'GUARDRAIL_VERSION', None)
    if guardrail_id and guardrail_version:
        return {'guardrail_id': guardrail_id, 'guardrail_version': guardrail_version}
    return {}


_AGENT_COLUMNS = ('inventory_agent', 'policy_agent', 'refund_agent', 'communication_agent')


def _context_summary(state: Optional[dict], limit: int = 1500) -> str:
    """Findings already stored in WorkflowState, handed to the next worker as context."""
    parts = []
    for col in _AGENT_COLUMNS:
        value = (state or {}).get(col)
        if value:
            parts.append(f"- {col}: {str(value)[:limit]}")
    return "\n".join(parts) if parts else "- (no findings stored yet)"


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    # Model: WORKER model (Claude Sonnet 4.5), low temperature for factual lookups
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,        # Claude Sonnet 4.5
        region_name=config.AWS_REGION,
        temperature=0.1,
        **_guardrail_kwargs(),
    )

    # System prompt for the Inventory Agent
    system_prompt = """You are the InventoryAgent for NovaMart customer support.

Your ONLY job is to gather FACTS about orders and customers from DynamoDB using your tools.
You never make decisions.

Tools:
- check_order_status(customer_id, order_id): look up one order (needs BOTH ids).
- get_customer_tier(customer_id): look up the customer's tier (Standard or Premium) and account details.
- list_customer_orders(customer_id): list all of a customer's orders.

Rules:
1. Extract the customer ID (e.g. CUST-001) and any order ID (e.g. ORD-27176) from the request.
2. For order status / return / refund requests, call BOTH check_order_status and get_customer_tier so
   the tier and the order facts are reported together.
3. For account questions (tier, membership, "am I premium?"), call get_customer_tier.
4. If the customer gives no order ID but asks about their orders, call list_customer_orders.
5. Report facts exactly as returned by the tools: order_id, status, product_name, order_date, price/amount,
   customer tier. Never invent data. If a record is not found, say so clearly.
6. Do NOT decide return or refund eligibility, do NOT quote policy, and do NOT write customer-facing prose.
   Return a concise, structured fact summary only."""

    # Implement check_order_status
    # NOTE: the Orders table has a COMPOSITE key (customer_id = partition key,
    # order_id = sort key), so a get_item needs BOTH values. That is why this
    # tool takes customer_id as well as order_id.
    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Look up one order in DynamoDB and report its status, product, dates
        and amount. Reports facts only - it does NOT decide return eligibility.

        Args:
            customer_id: The customer's unique identifier (e.g. CUST-001)
            order_id: The order identifier (e.g. ORD-27176)

        Returns:
            Order record (order_id, status, product_name, order_date, price, ...)
            or a not-found message
        """
        try:
            response = _orders_table().get_item(
                Key={'customer_id': customer_id, 'order_id': order_id}
            )
        except ClientError as exc:
            return {'found': False, 'error': str(exc)}
        item = response.get('Item')
        if not item:
            return {
                'found': False,
                'message': f"No order {order_id} found for customer {customer_id}.",
            }
        result = _json_safe(item)
        result['found'] = True
        return result

    # Implement get_customer_tier
    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        try:
            response = _customers_table().get_item(Key={'customer_id': customer_id})
        except ClientError as exc:
            return {'found': False, 'error': str(exc)}
        item = response.get('Item')
        if not item:
            return {'found': False, 'message': f"No customer {customer_id} found."}
        result = _json_safe(item)
        result['found'] = True
        return result

    # Implement list_customer_orders
    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        try:
            table = _orders_table()
            kwargs = {'KeyConditionExpression': Key('customer_id').eq(customer_id)}
            items = []
            while True:
                response = table.query(**kwargs)
                items.extend(response.get('Items', []))
                last_key = response.get('LastEvaluatedKey')
                if not last_key:
                    break
                kwargs['ExclusiveStartKey'] = last_key
        except ClientError as exc:
            return {'found': False, 'error': str(exc)}
        orders = _json_safe(items)
        return {
            'customer_id': customer_id,
            'order_count': len(orders),
            'orders':      orders,
        }

    # Instantiate and return the Agent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[check_order_status, get_customer_tier, list_customer_orders],
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    # Model: WORKER model, low temperature for consistent decisions
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,        # Claude Sonnet 4.5
        region_name=config.AWS_REGION,
        temperature=0.1,
        **_guardrail_kwargs(),
    )

    # System prompt for the Refund Agent
    system_prompt = """You are the RefundAgent for NovaMart customer support.

You make return/refund ELIGIBILITY DECISIONS using facts already gathered by the InventoryAgent.

Tools:
- get_inventory_context(session_id): read the facts gathered by the InventoryAgent from WorkflowState.
- initiate_refund(customer_id, order_id, reason): start the return for an eligible order.

Process:
1. Call get_inventory_context with the session ID to read the order facts and the customer tier.
2. Apply the return window by customer tier, counted in days from the order date to today's date
   (today's date is provided in the request):
     - Standard customers: 30 days
     - Premium customers:  60 days
3. The order is ELIGIBLE only if (a) it is within the window for the customer's tier and
   (b) its status shows it can be returned (for example it was delivered and has not already been
   returned or cancelled).
4. If ELIGIBLE: call initiate_refund(customer_id, order_id, reason) using the customer's stated
   reason (or "Customer requested return" if none was given), then report the decision and the
   return_reference number it gives you.
5. If NOT eligible: do NOT call initiate_refund. Explain exactly why (days since order, window for
   the tier, or status) and state that the refund is denied.
6. If the inventory context is missing or incomplete, say what is missing instead of guessing.

State your decision clearly (APPROVED or DENIED) with the supporting numbers: order date, days elapsed,
tier, allowed window. Do not write customer-facing prose; a concise decision summary is enough."""

    # Implement get_inventory_context
    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        state = _read_workflow_state(session_id) or {}
        context = state.get('inventory_agent')
        if not context:
            return {}
        return {'inventory_agent': _json_safe(context)}

    # Implement initiate_refund
    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        return_reference = f"RET-{uuid.uuid4().hex[:8].upper()}"
        initiated_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        try:
            _orders_table().update_item(
                Key={'customer_id': customer_id, 'order_id': order_id},
                UpdateExpression=(
                    "SET #s = :status, return_reference = :ref, "
                    "return_reason = :reason, return_initiated_at = :ts"
                ),
                ConditionExpression='attribute_exists(order_id)',
                ExpressionAttributeNames={'#s': 'status'},
                ExpressionAttributeValues={
                    ':status': 'RETURN_INITIATED',
                    ':ref':    return_reference,
                    ':reason': reason,
                    ':ts':     initiated_at,
                },
            )
        except ClientError as exc:
            if exc.response['Error']['Code'] == 'ConditionalCheckFailedException':
                return {
                    'success': False,
                    'message': f"Order {order_id} not found for customer {customer_id}.",
                }
            return {'success': False, 'error': str(exc)}
        return {
            'success':          True,
            'customer_id':      customer_id,
            'order_id':         order_id,
            'return_reference': return_reference,
            'status':           'RETURN_INITIATED',
            'reason':           reason,
            'instructions': (
                "Pack the item in its original packaging, include the return reference "
                f"{return_reference}, and use the prepaid return label sent by email. "
                "The refund is issued to the original payment method once the item is received."
            ),
        }

    # Instantiate and return the Agent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_inventory_context, initiate_refund],
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    retriever_prompt_tpl = (
        "You are the {domain}PolicyRetriever for NovaMart. Always call your retrieval tool "
        "with the user's question, then return the relevant passages from the {domain_lower} "
        "policy knowledge base as faithfully as possible, keeping any source names and "
        "relevance scores. Do not invent information. If nothing relevant is found, "
        "reply exactly: No relevant {domain_lower} policy information found."
    )

    def _kb_search(kb_id: str, label: str, query: str) -> str:
        if not kb_id:
            return f"[{label} knowledge base is not configured]"
        try:
            results = retrieve_from_knowledge_base(kb_id, query)
            return format_kb_results(results)
        except Exception as exc:  # surfaced to the agent instead of crashing the graph
            return f"[{label} knowledge base retrieval failed: {exc}]"

    def _build_retriever(domain: str, retrieval_tool) -> Agent:
        return Agent(
            model=BedrockModel(
                model_id=config.WORKER_MODEL_ID,        # Claude Sonnet 4.5
                region_name=config.AWS_REGION,
                temperature=0.1,
                **_guardrail_kwargs(),
            ),
            system_prompt=retriever_prompt_tpl.format(domain=domain, domain_lower=domain.lower()),
            tools=[retrieval_tool],
            callback_handler=None,
        )

    # Build ReturnsPolicyRetrieverAgent
    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        return _kb_search(config.RETURNS_KB_ID, 'Returns', query)

    # Create the ReturnsPolicyRetrieverAgent with the tool above
    returns_retriever = _build_retriever('Returns', retrieve_returns_policy)

    # Build ShippingPolicyRetrieverAgent
    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        return _kb_search(config.SHIPPING_KB_ID, 'Shipping', query)

    # Create the ShippingPolicyRetrieverAgent with the tool above
    shipping_retriever = _build_retriever('Shipping', retrieve_shipping_policy)

    # Build WarrantyPolicyRetrieverAgent
    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        return _kb_search(config.WARRANTY_KB_ID, 'Warranty', query)

    # Create the WarrantyPolicyRetrieverAgent with the tool above
    warranty_retriever = _build_retriever('Warranty', retrieve_warranty_policy)

    _score_re = re.compile(r'(?i)(?:relevance|score)[^0-9\-]{0,12}(-?[0-9]*\.?[0-9]+)')

    def _merge_results(results: dict) -> str:
        """Merge per-domain results: dedupe identical passages, rank by relevance when scores exist."""
        passages, seen = [], set()
        order = 0
        for domain in ['Returns', 'Shipping', 'Warranty']:
            text = results.get(domain, '') or ''
            for block in re.split(r'\n\s*\n', text):
                block = block.strip()
                if not block:
                    continue
                key = re.sub(r'\W+', ' ', block).strip().lower()
                if key in seen or key.startswith('no relevant'):
                    continue
                seen.add(key)
                match = _score_re.search(block)
                score = float(match.group(1)) if match else None
                passages.append((domain, block, score, order))
                order += 1
        # Highest relevance first; passages without a score keep their original order after scored ones
        passages.sort(key=lambda p: (p[2] is None, -(p[2] or 0.0), p[3]))
        if not passages:
            return "No relevant policy information found in the Returns, Shipping or Warranty knowledge bases."
        return "\n\n".join(f"[{domain} policy]\n{block}" for domain, block, _, _ in passages)

    # Implement search_all_policies - parallel RAG retrieval tool
    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """
        # Build a dict mapping domain names to their retriever agents
        retrievers = {
            'Returns':  returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever,
        }

        # ── Trace: show parallel KB dispatch to learners ──────────────────
        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        # Define a helper to run one retriever sub-agent
        def _run_retriever(domain: str, agent, query: str) -> tuple:
            """
            Run one retriever sub-agent and return (domain, result_text).

            stdout is suppressed globally for all threads by the
            _TraceWriter._suppress_parallel flag set in kb_start().
            This covers both the direct worker thread and any internal
            streaming child threads that Strands SDK spawns internally -
            which do NOT inherit thread-local variables and therefore cannot
            be suppressed with a thread-local capture approach.
            Results are returned as values and printed cleanly and
            sequentially by trace.kb_result() after all futures join.
            """
            try:
                # Each retriever is stateless between searches: drop old turns
                messages = getattr(agent, 'messages', None)
                if isinstance(messages, list):
                    messages.clear()
                response = agent(
                    f"Search the {domain} policy knowledge base for: {query}"
                )
                return domain, str(response)
            except Exception as exc:
                logger.warning(f"{domain} retriever failed: {exc}")
                return domain, f"[{domain} retrieval error: {exc}]"

        # Use ThreadPoolExecutor to run all three retrievers in parallel
        # Collect results into a dict: {'Returns': '...', 'Shipping': '...', ...}
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_run_retriever, domain, agent, query): domain
                for domain, agent in retrievers.items()
            }
            for future in as_completed(futures):
                domain = futures[future]
                try:
                    _, text = future.result()
                except Exception as exc:
                    text = f"[{domain} retrieval error: {exc}]"
                results[domain] = text

        # ── Trace: all KBs responded - print each result sequentially ─────
        trace.kb_done(len(retrievers))
        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        # Combine results from all three domains and return
        return _merge_results(results)

    # Model for the PolicyAgent coordinator
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,        # Claude Sonnet 4.5
        region_name=config.AWS_REGION,
        temperature=0.2,
        **_guardrail_kwargs(),
    )

    # System prompt for PolicyAgent coordinator
    system_prompt = """You are the PolicyAgent for NovaMart customer support - a multi-agent RAG coordinator.

You answer questions about NovaMart's returns, shipping and warranty policies using ONLY
information retrieved from the policy knowledge bases.

Process:
1. ALWAYS call search_all_policies exactly once with the customer's question. It queries the Returns,
   Shipping and Warranty knowledge bases in parallel and returns merged, de-duplicated passages.
2. Synthesize the passages into a complete, accurate and grounded answer. Cover the details that matter
   (time windows, tiers such as Standard 30 days vs Premium 60 days, rates, fees, conditions, exceptions).
3. Name which policy area (Returns, Shipping, Warranty) each key point comes from.
4. If the retrieved passages do not answer the question, say so plainly. Never invent policy terms.
5. You only know policy text - you have no access to customer or order data. Do not guess a specific
   customer's tier, orders or eligibility.
6. Return a factual policy summary, not a final customer-facing message."""

    # Instantiate and return the PolicyAgent coordinator
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[search_all_policies],
    )


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────

def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """

    # Model: WORKER model, slightly higher temperature for natural tone
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,        # Claude Sonnet 4.5
        region_name=config.AWS_REGION,
        temperature=0.3,
        **_guardrail_kwargs(),
    )

    # System prompt for the Communication Agent
    system_prompt = """You are the CommunicationAgent for NovaMart customer support.

You write the FINAL customer-facing response. You are always the last agent to run.

Process:
1. Call get_full_workflow_context with the session ID to read everything the other agents found
   (inventory_agent, policy_agent, refund_agent).
2. Compose one clear, warm, empathetic and professional reply to the customer's original request using
   ONLY that information. Do not invent order details, policies, dates or reference numbers.
3. Refund/return requests: state the decision (approved or denied) and the reason. If approved, include
   the return reference number and next steps. If denied, explain why kindly and offer alternatives.
4. Policy questions: give the policy details clearly (windows, tiers, conditions) as found by the PolicyAgent.
5. Account questions: answer from the facts the InventoryAgent found (for example the customer's tier).
6. Math / calculation questions (no other agent data is needed): calculate the answer yourself, working
   carefully with full precision - for example quantity x unit price, then apply the discount - and round
   currency to two decimals (half up) ONLY at the very end, after the full calculation. Show the steps briefly.
7. Keep the tone friendly and concise, address the customer directly, and end by offering further help.
8. Never mention internal agents, tools, sessions, databases or workflow state. Output only the message
   the customer should read."""

    # Implement get_full_workflow_context
    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent)
        """
        state = _read_workflow_state(session_id)
        if not state:
            return {}
        return _json_safe(state)

    # Instantiate and return the Agent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context],
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    # Model: ORCHESTRATOR model (Claude Haiku 4.5), deterministic routing
    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.0,
        **_guardrail_kwargs(),
    )

    # System prompt for the Orchestrator
    # For arithmetic, skip Inventory, Policy and Refund, but still call
    # CommunicationAgent last. Round currency only after the full calculation.
    system_prompt = """You are the OrchestratorAgent for NovaMart customer support.
You ROUTE requests to specialist agents using your tools. You NEVER answer the customer yourself.

Every request starts with: [Session ID: <id>] [Customer ID: <id>] <customer message>.
Use those exact session and customer IDs in every tool call.

You MUST follow these six routing rules EXACTLY. They are explicit instructions, not suggestions:

RULE 1 - Every request, always: call initialize_session FIRST, before any other tool.

RULE 2 - Order status, return or refund requests: call route_to_inventory_agent, and THEN call
         route_to_refund_agent (always in that order, never refund before inventory).

RULE 3 - Policy meaning questions (return windows, shipping rates, warranty terms, conditions):
         call route_to_policy_agent.

RULE 4 - Account questions ("what is my tier?", "am I premium?", "what orders do I have?"):
         call route_to_inventory_agent. NEVER call route_to_policy_agent for an account question:
         the PolicyAgent only knows policy text, it does NOT know customer data such as tiers or orders.
         Account questions go to InventoryAgent, never to PolicyAgent.

RULE 5 - Math / calculation questions (prices, quantities, discounts): calculate WITHOUT calling
         route_to_inventory_agent, route_to_policy_agent or route_to_refund_agent. Work out the full
         calculation first and round currency only AFTER the full calculation is complete. Then call
         route_to_communication_agent LAST, passing the customer's original message.

RULE 6 - Every request, always, as your VERY LAST step: call route_to_communication_agent to compose
         the final customer reply.

CRITICAL: You are NEVER permitted to write the final customer-facing response yourself. You must ALWAYS
delegate to route_to_communication_agent as your very last action - no exceptions, even when you believe
you already have a complete answer. After route_to_communication_agent returns, output exactly the text
it returned, unchanged, and add nothing else.

Pass the customer's message verbatim in the request / original_request arguments."""

    # Each routing tool follows the same lifecycle:
    #   1. read the current WorkflowState and remember its version
    #   2. invoke the worker agent (with the current context)
    #   3. write the result back with _update_workflow_state(), checked against the
    #      version read in step 1 (optimistic locking)
    #   4. trace.step_start(...) before the worker, trace.step_done(..., old_version) after

    # Implement route_to_inventory_agent
    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        # 1. read WorkflowState (and the version the update will be checked against)
        state = _read_workflow_state(session_id) or _ensure_workflow_state(session_id, customer_id)
        old_version = int(state['version'])
        # 2. call the worker with current context
        prompt = (f"[Session ID: {session_id}] [Customer ID: {customer_id}]\n"
                  f"Current WorkflowState findings:\n{_context_summary(state)}\n\n"
                  f"Gather the order and customer facts needed for this request: {request}")
        trace.step_start('inventory_agent')
        try:
            text = str(inventory_agent(prompt))
        except Exception as exc:
            logger.error(f"inventory_agent failed: {exc}")
            text = f"[inventory_agent error: {exc}]"
        # 3. persist with optimistic locking
        _update_workflow_state(session_id, {'inventory_agent': text},
                               expected_version=old_version)
        trace.step_done('inventory_agent', old_version)
        return text

    # Implement route_to_policy_agent
    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        state = _read_workflow_state(session_id) or _ensure_workflow_state(session_id)
        old_version = int(state['version'])
        prompt = (f"[Session ID: {session_id}]\n"
                  f"Current WorkflowState findings:\n{_context_summary(state)}\n\n"
                  f"Answer this policy question using the policy knowledge bases: {request}")
        trace.step_start('policy_agent')
        try:
            text = str(policy_agent(prompt))
        except Exception as exc:
            logger.error(f"policy_agent failed: {exc}")
            text = f"[policy_agent error: {exc}]"
        _update_workflow_state(session_id, {'policy_agent': text},
                               expected_version=old_version)
        trace.step_done('policy_agent', old_version)
        return text

    # Implement route_to_refund_agent
    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        state = _read_workflow_state(session_id) or _ensure_workflow_state(session_id, customer_id)
        old_version = int(state['version'])
        today = time.strftime('%Y-%m-%d', time.gmtime())
        prompt = (f"[Session ID: {session_id}] [Customer ID: {customer_id}] "
                  f"Today's date (UTC): {today}.\n"
                  f"Current WorkflowState findings:\n{_context_summary(state)}\n\n"
                  f"Decide eligibility and, if eligible, initiate the return for this request: {request}")
        trace.step_start('refund_agent')
        try:
            text = str(refund_agent(prompt))
        except Exception as exc:
            logger.error(f"refund_agent failed: {exc}")
            text = f"[refund_agent error: {exc}]"
        _update_workflow_state(session_id, {'refund_agent': text},
                               expected_version=old_version)
        trace.step_done('refund_agent', old_version)
        return text

    # Implement route_to_communication_agent
    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        state = _read_workflow_state(session_id) or _ensure_workflow_state(session_id, customer_id)
        old_version = int(state['version'])
        prompt = (f"[Session ID: {session_id}] [Customer ID: {customer_id}]\n"
                  f"Current WorkflowState findings:\n{_context_summary(state)}\n\n"
                  f"Compose the final customer-facing response to this original request: "
                  f"{original_request}")
        trace.step_start('communication_agent')
        try:
            text = str(communication_agent(prompt))
        except Exception as exc:
            logger.error(f"communication_agent failed: {exc}")
            text = f"[communication_agent error: {exc}]"
        _update_workflow_state(session_id, {'communication_agent': text},
                               expected_version=old_version)
        trace.step_done('communication_agent', old_version)
        return text

    # Implement initialize_session
    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        try:
            _create_workflow_state(session_id, customer_id)
        except ClientError as exc:
            if exc.response['Error']['Code'] == 'ConditionalCheckFailedException':
                return (f"Session {session_id} already initialized for customer {customer_id}. "
                        f"Continue routing.")
            raise
        return f"Session {session_id} initialized for customer {customer_id}."

    # Instantiate and return the OrchestratorAgent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
    )


# ═══════════════════════════════════════════════════════
#  AGENT GRAPH HELPERS
# ═══════════════════════════════════════════════════════

def _apply_guardrail(agents: list) -> None:
    """
    Attach the Bedrock Guardrail (Task 3) to every agent's BedrockModel.
    Guardrails are enforced per model invocation, so once GUARDRAIL_ID /
    GUARDRAIL_VERSION are known (in .env locally, as runtime environment
    variables when deployed) every agent in the graph runs behind the
    guardrail - no change to the agents themselves is needed.
    """
    guardrail_id      = config.GUARDRAIL_ID
    guardrail_version = config.GUARDRAIL_VERSION
    if not guardrail_id or not guardrail_version:
        return
    for agent in agents:
        model = getattr(agent, 'model', None)
        if model is not None and hasattr(model, 'update_config'):
            model.update_config(guardrail_id=guardrail_id,
                                guardrail_version=guardrail_version)


def build_agent_graph(verbose: bool = False) -> Agent:
    """Build all five agents, apply the guardrail, return the orchestrator."""
    def _ok(label):
        if verbose:
            print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  {label}{_C.RESET}", flush=True)

    inventory_agent     = build_inventory_agent();     _ok('InventoryAgent')
    refund_agent        = build_refund_agent();        _ok('RefundAgent')
    policy_agent        = build_policy_agent();        _ok('PolicyAgent')
    communication_agent = build_communication_agent(); _ok('CommunicationAgent')
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    _ok('Orchestrator')
    _apply_guardrail([inventory_agent, refund_agent, policy_agent,
                      communication_agent, orchestrator])
    if verbose and config.GUARDRAIL_ID:
        print(f"  {_C.GRY}          Guardrail {config.GUARDRAIL_ID} "
              f"(v{config.GUARDRAIL_VERSION}) attached to all agents{_C.RESET}")
    return orchestrator


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT TOOLING - AgentCore CLI
#
#  The runtime is deployed with the AgentCore CLI (`agentcore`, npm package
#  @aws/agentcore - https://github.com/aws/agentcore-cli) through the
#  pre-written helper src/agentcore_cli.py:
#
#    agentcore_cli.stage_runtime_code()   copies this file, its helper modules
#                                         and config.py to build/runtime/ with a
#                                         pyproject.toml of the runtime deps
#    agentcore_cli.configure_runtime()    writes the runtime settings (network
#                                         mode, protocol, execution role, env
#                                         vars) to agentcore/agentcore.json
#    agentcore_cli.deploy()               runs `agentcore deploy -y`: the CLI
#                                         downloads arm64 / Python 3.12 wheels
#                                         with uv, zips them with the code
#                                         (direct code deployment) and creates
#                                         or updates the runtime via CDK
#    agentcore_cli.deployed_runtime_arn() reads the ARN the CLI recorded
#
#  Inside the runtime this same file is the entry point: it is started with
#  no command-line argument and serves HTTP (see run_serve). The marker file
#  written next to it by stage_runtime_code() tells __main__ to do so.
# ═══════════════════════════════════════════════════════

_RUNTIME_MARKER = '.agentcore-runtime'         # written by agentcore_cli.stage_runtime_code()
_SRC_DIR        = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version) - the version is always a number, never DRAFT.
    If a guardrail with the project name already exists and matches the required policies it is
    reused; if it is stale (e.g. from an earlier attempt) it is updated and re-versioned.
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    blocked_message = (
        "Sorry, I can't help with that request. I can assist with NovaMart orders, "
        "returns, shipping, warranty and account questions."
    )

    # Content filters: SEXUAL / VIOLENCE / HATE at HIGH, INSULTS / MISCONDUCT at MEDIUM
    filters_config = [
        {'type': 'SEXUAL',     'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
        {'type': 'VIOLENCE',   'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
        {'type': 'HATE',       'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
        {'type': 'INSULTS',    'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
        {'type': 'MISCONDUCT', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
    ]

    # PII: block cards and SSNs, anonymize emails and phone numbers
    pii_entities = [
        {'type': 'CREDIT_DEBIT_CARD_NUMBER',  'action': 'BLOCK'},
        {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
        {'type': 'EMAIL',                     'action': 'ANONYMIZE'},
        {'type': 'PHONE',                     'action': 'ANONYMIZE'},
    ]

    # Denied topics (one DENY topic per blocked area, STANDARD tier definitions).
    # Pricing negotiation is defined narrowly (haggling / changing an advertised price)
    # and explicitly excludes arithmetic with an already-specified price and discount.
    topics_config = [
        {
            'name': 'competitor-products',
            'definition': (
                "Requests to compare NovaMart with, recommend, or promote competitor "
                "companies, their products, or their prices."
            ),
            'examples': [
                "Is Amazon cheaper than NovaMart?",
                "Should I buy this from Best Buy instead?",
                "Which competitor store has better laptops than yours?",
            ],
            'type': 'DENY',
        },
        {
            'name': 'pricing-negotiation',
            'definition': (
                "Attempts to haggle, bargain, or ask NovaMart to change, lower, or override an "
                "advertised price or grant an unadvertised discount. Does NOT include arithmetic "
                "using an already-specified price, quantity, and discount, which is allowed."
            ),
            'examples': [
                "Can you give me a better price if I buy today?",
                "I'll pay half of the listed price, deal?",
                "Lower the advertised price for me or I'll shop elsewhere.",
            ],
            'type': 'DENY',
        },
        {
            'name': 'legal-threats',
            'definition': (
                "Threats of lawsuits or legal action against NovaMart, or requests for legal "
                "advice about suing or legal claims."
            ),
            'examples': [
                "I'm going to sue you if you don't refund me.",
                "My lawyer will contact you about this order.",
                "Can you advise me how to file a legal claim against NovaMart?",
            ],
            'type': 'DENY',
        },
    ]

    guardrail_params = dict(
        name=config.GUARDRAIL_NAME,
        description=(
            "NovaMart enterprise safety guardrail: harmful content filters, PII protection, "
            "denied topics and profanity filtering."
        ),
        contentPolicyConfig={'filtersConfig': filters_config},
        sensitiveInformationPolicyConfig={'piiEntitiesConfig': pii_entities},
        topicPolicyConfig={
            'topicsConfig': topics_config,
            'tierConfig': {'tierName': 'STANDARD'},
        },
        wordPolicyConfig={'managedWordListsConfig': [{'type': 'PROFANITY'}]},
        blockedInputMessaging=blocked_message,
        blockedOutputsMessaging=blocked_message,
        crossRegionConfig={'guardrailProfileIdentifier': 'us.guardrail.v1:0'},
    )

    def _numbered_versions(gid: str) -> list:
        versions, token = [], None
        while True:
            kwargs = {'guardrailIdentifier': gid}
            if token:
                kwargs['nextToken'] = token
            page = bedrock_client.list_guardrails(**kwargs)
            for g in page.get('guardrails', []):
                v = str(g.get('version', 'DRAFT'))
                if v.isdigit():
                    versions.append(int(v))
            token = page.get('nextToken')
            if not token:
                return sorted(versions)

    def _matches_required(gid: str) -> bool:
        """True when the DRAFT already has every required policy at the required strength/action."""
        d = bedrock_client.get_guardrail(guardrailIdentifier=gid, guardrailVersion='DRAFT')
        filters = {f['type']: (f.get('inputStrength'), f.get('outputStrength'))
                   for f in d.get('contentPolicy', {}).get('filters', [])}
        for f in filters_config:
            if filters.get(f['type']) != (f['inputStrength'], f['outputStrength']):
                return False
        pii = {e['type']: e.get('action')
               for e in d.get('sensitiveInformationPolicy', {}).get('piiEntities', [])}
        for e in pii_entities:
            if pii.get(e['type']) != e['action']:
                return False
        topics = {t['name'] for t in d.get('topicPolicy', {}).get('topics', [])}
        if not {t['name'] for t in topics_config} <= topics:
            return False
        words = {w['type'] for w in d.get('wordPolicy', {}).get('managedWordLists', [])}
        return 'PROFANITY' in words

    def _wait_ready(gid: str, timeout: int = 90) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = bedrock_client.get_guardrail(
                guardrailIdentifier=gid, guardrailVersion='DRAFT').get('status')
            if status == 'READY':
                return
            if status == 'FAILED':
                raise RuntimeError(f"Guardrail {gid} entered FAILED status")
            time.sleep(3)

    # Reuse an existing guardrail when it is correct; otherwise fix it in place
    existing_id, token = None, None
    while existing_id is None:
        kwargs = {'nextToken': token} if token else {}
        page = bedrock_client.list_guardrails(**kwargs)
        for g in page.get('guardrails', []):
            if g['name'] == config.GUARDRAIL_NAME:
                existing_id = g['id']
                break
        token = page.get('nextToken')
        if not token:
            break

    if existing_id:
        versions = _numbered_versions(existing_id)
        if versions and _matches_required(existing_id):
            print(f"Guardrail already exists: {existing_id} (version: {versions[-1]})")
            return existing_id, str(versions[-1])
        print(f"Guardrail {existing_id} is missing required settings - updating it")
        bedrock_client.update_guardrail(guardrailIdentifier=existing_id, **guardrail_params)
        guardrail_id = existing_id
    else:
        response = bedrock_client.create_guardrail(**guardrail_params)
        guardrail_id = response['guardrailId']

    _wait_ready(guardrail_id)

    # Promote from DRAFT to a numbered version
    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description='Production version',
    )
    guardrail_version = str(version_response['version'])
    print(f"Guardrail ready: {guardrail_id} (version: {guardrail_version})")
    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime with
    the AgentCore CLI (src/agentcore_cli.py wraps it).

    AgentCore does not serialize Python objects, so `orchestrator_agent` is
    not uploaded directly. Instead the pre-written staging step copies this
    file, which doubles as the HTTP entry point (see run_serve), together
    with its helper modules and config.py to build/runtime/. `agentcore
    deploy` then packages that directory with arm64 dependencies and creates
    or updates the runtime ("direct code deployment"). Re-running is safe:
    an unchanged runtime is left alone, a changed one is updated in place.

    The guardrail is attached by environment variables: inside the runtime
    build_agent_graph() reads GUARDRAIL_ID / GUARDRAIL_VERSION and applies
    them to every agent's model (see _apply_guardrail), exactly as `test`
    and `chat` do locally.

    Returns:
        The AgentCore Runtime ARN
    """
    import agentcore_cli

    runtime_name = config.AGENTCORE_RUNTIME_NAME
    print(f"  AWS Account: {config.ACCOUNT_ID}  |  Region: {config.AWS_REGION}")
    print(f"  Runtime: {runtime_name}  |  CLI project: agentcore/agentcore.json "
          f"(stack {agentcore_cli.stack_name()})")
    previous_arn = agentcore_cli.deployed_runtime_arn()
    if previous_arn:
        print(f"  Runtime already deployed - updating it: {previous_arn}")

    # Stage the code the CLI packages (src modules + config.py + pyproject.toml).
    agentcore_cli.stage_runtime_code()

    # The runtime must receive a NUMBERED guardrail version (never DRAFT)
    if not guardrail_id or not str(guardrail_version).isdigit():
        raise ValueError(
            f"Guardrail must have a numbered version before deploying "
            f"(got id={guardrail_id!r}, version={guardrail_version!r})"
        )

    # 1. Runtime environment variables
    runtime_env = {
        'AWS_REGION':         config.AWS_REGION,
        'PROJECT_NAME':       config.PROJECT_NAME,
        'RETURNS_KB_ID':      config.RETURNS_KB_ID or '',
        'SHIPPING_KB_ID':     config.SHIPPING_KB_ID or '',
        'WARRANTY_KB_ID':     config.WARRANTY_KB_ID or '',
        'AGENT_LOG_GROUP':    config.AGENT_LOG_GROUP,
        'GUARDRAIL_ID':       guardrail_id,
        'GUARDRAIL_VERSION':  guardrail_version,
    }

    # 2. Write the runtime settings to agentcore/agentcore.json
    agentcore_cli.configure_runtime(
        env_vars=runtime_env,
        network_mode='PUBLIC',
        protocol='HTTP',
        execution_role_arn=config.AGENTCORE_ROLE_ARN,
    )

    # 3. Deploy (runs `agentcore deploy -y`)
    agentcore_cli.deploy()

    # 4. Read the ARN the CLI recorded
    runtime_arn = agentcore_cli.deployed_runtime_arn()

    if not runtime_arn:
        raise RuntimeError("agentcore deploy finished but no runtime ARN was recorded "
                           "(check `agentcore status` and agentcore/.cli/logs/)")

    # Wait for the runtime to become READY and return its ARN.
    print(f"  Runtime deployed: {runtime_arn}")
    print("  Waiting for runtime status READY", end='', flush=True)
    wait_for_runtime_ready(agentcore_control, runtime_arn.split('/')[-1])
    print(' ready.')
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Create an AgentCore Memory resource for session-scoped conversational
    context. Uses the SESSION_SUMMARY (summaryMemoryStrategy) strategy with
    7-day event retention.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAME
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m.get('arn') or m.get('memoryArn')
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    # Create AgentCore Memory (SESSION_SUMMARY strategy, 7-day event expiry)
    response = agentcore_control.create_memory(
        name=memory_name,
        description=(
            "NovaMart session memory: keeps conversation context across turns "
            "so customers do not have to repeat themselves."
        ),
        eventExpiryDuration=7,
        memoryStrategies=[{
            'summaryMemoryStrategy': {
                'name': 'SessionSummary',
                'namespaces': ['/summaries/{actorId}/{sessionId}'],
            }
        }],
        clientToken=str(uuid.uuid4()),
    )

    if response is None:
        raise NotImplementedError("configure_memory: create_memory() not implemented")

    # The create_memory response nests the resource under 'memory'; its ARN is the 'arn'
    # field (the service's memoryArn). Fall back to top-level / alternate spellings.
    memory = response.get('memory', response)

    def _arn_of(resource: dict) -> Optional[str]:
        return resource.get('arn') or resource.get('memoryArn')

    memory_arn = _arn_of(memory) or response.get('memoryArn')
    if not memory_arn:
        raise RuntimeError(f"create_memory returned no memory ARN: {response}")
    print(f"  Memory created: {memory_arn}  (status: {memory.get('status')})")

    # Wait until the memory resource is ACTIVE, then return its ARN.
    print("  Waiting for memory status ACTIVE", end='', flush=True)
    deadline = time.time() + 300
    while memory.get('status') != 'ACTIVE' and time.time() < deadline:
        time.sleep(10)
        print('.', end='', flush=True)
        memory = agentcore_control.get_memory(memoryId=memory['id'])['memory']
        memory_arn = _arn_of(memory) or memory_arn
        if memory.get('status') == 'FAILED':
            raise RuntimeError(f"Memory creation failed: {memory.get('failureReason')}")
    print(' ready.' if memory.get('status') == 'ACTIVE' else f" status {memory.get('status')}")
    return memory_arn


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure observability for the deployed agent:
    - Agent logs → CloudWatch Logs at INFO level (config.AGENT_LOG_GROUP)
    - Execution traces → AWS X-Ray at 100% sampling

    The loggingConfiguration built here is applied by
    apply_observability_config() (agent_observability.py):
      cloudWatchConfig -> log group created; runtime env AGENT_LOG_GROUP /
                          AGENT_LOG_LEVEL so the deployed agent ships its logs there
      xRayConfig       -> CloudWatch Transaction Search enabled with the given
                          sampling percentage; runtime env AGENT_TRACING_ENABLED /
                          AGENT_TRACE_SAMPLING_RATE
    """
    logging_configuration = {
        'cloudWatchConfig': {
            'logGroupName': config.AGENT_LOG_GROUP,
            'logLevel':     'INFO',
            'enabled':      True,
        },
        'xRayConfig': {
            'enabled':      True,
            'samplingRate': 1.0,
        },
    }
    try:
        # The completed loggingConfiguration is handed to the pre-written helper
        summary = apply_observability_config(runtime_arn, logging_configuration)
        print(f"  CloudWatch logs : {logging_configuration['cloudWatchConfig']['logGroupName']}")
        print(f"  X-Ray sampling  : {logging_configuration['xRayConfig']['samplingRate']:.0%}")
        if summary:
            logger.info(f"Observability configuration applied: {summary}")
    except Exception as e:
        print(f"[Note] Observability configuration failed: {e}")


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Deployment pattern:
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION = os.environ.get('ORDERS_FUNCTION', '')
_POLICY_FUNCTION = os.environ.get('POLICY_FUNCTION', '')
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', '')


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Optional extension to the in-process @tool functions. Resolve configured
    Lambda functions first; if none exist, skip gateway creation. Otherwise
    create/reuse the gateway and submit its targets. Connecting agents to this
    MCP endpoint requires separate integration; this starter uses in-process tools.

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        A SKIPPED result with a reason, or gateway details and target count.
    """
    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    # Resolve optional Lambda targets before creating any gateway resources.
    available = []
    for target in targets:
        if not target['function']:
            continue
        try:
            available.append((target, _gw_get_function_arn(target['function'])))
        except ClientError as exc:
            if exc.response['Error']['Code'] != 'ResourceNotFoundException':
                raise
            print(f"    [Skipped] {target['name']}: Lambda function not found")

    if not available:
        return {'status': 'SKIPPED', 'reason': 'No configured Lambda tool functions are available.'}

    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    print(f"\n  Registering {len(available)} Gateway targets...")
    for target, lambda_arn in available:
        _gw_create_target(agentcore_ctrl, gateway_id, target, lambda_arn)

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url,
            'status': 'TARGETS_SUBMITTED', 'target_count': len(available)}



# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> dict:
    """
    Invoke the deployed agent via AgentCore Runtime (see run_serve).

    AgentCore requires runtimeSessionId to be at least 33 characters, so the
    short project session id is embedded in a longer, unique runtime session id.
    """
    if not config.AGENTCORE_RUNTIME_ARN:
        raise RuntimeError("AGENTCORE_RUNTIME_ARN is not set - run the deploy command first")

    runtime_session_id = f"{session_id}-{uuid.uuid4().hex}"     # >= 33 chars
    payload = json.dumps({
        'prompt':      user_message,
        'session_id':  session_id,
        'customer_id': customer_id,
    })
    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        runtimeSessionId=runtime_session_id,
        contentType='application/json',
        accept='application/json',
        payload=payload,
    )
    body = response['response'].read()
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return {'result': body.decode('utf-8', errors='replace') if isinstance(body, bytes) else str(body)}


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    # Fail fast if the AgentCore CLI (used by Steps 3 and 5) is missing.
    import agentcore_cli
    print(f"AgentCore CLI: {agentcore_cli.cli_version()} ({agentcore_cli.cli_path()})\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        if gw['status'] == 'SKIPPED':
            print(f"  [Skipped] Gateway: {gw['reason']}")
        else:
            print(f"  Gateway URL : {gw['gateway_url']}")
            print("  Lambda targets submitted; connect an MCP client separately to use them.")
    except Exception as e:
        print(f"  [Note] Optional Gateway deployment failed: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    print(f"  Then try the deployed runtime:")
    print(f"  python src/agent_orchestrator.py invoke \"What is the return policy for premium customers?\"")
    print(f"  or with the CLI:  agentcore invoke \"What is the return policy for premium customers?\"")
    print(f"  (agentcore status / agentcore logs show the deployed runtime and its logs)\n")
    return runtime_arn, guardrail_id


# ═══════════════════════════════════════════════════════
#  LOCAL TEST SCENARIOS
# ═══════════════════════════════════════════════════════

# Order IDs match infrastructure/seed_data.py.
TEST_CASES = [
    ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
    ("CUST-002", "What is the return policy for premium customers?"),
    ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
]

# Test customers shown by the chat command. Data matches seed_data.py.
TEST_CUSTOMERS = [
    ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Wireless Headphones Pro"),
    ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "Mechanical Keyboard K2"),
    ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "Laptop UltraBook 14"),
    ("CUST-004", "David Lee",     "Standard", "ORD-30001", "Phone Case Slim"),
]


def run_test_scenarios() -> None:
    """Run the three scenarios locally; every request is traced to X-Ray."""
    print("Running local agent test...")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph()

    for customer_id, query in TEST_CASES:
        session_id = str(uuid.uuid4())[:8]
        print(f"\n{'─'*60}")
        print(f"Session: {session_id} | Customer: {customer_id}")
        print(f"Query: {query}")
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
        with tracer.trace_request(session_id, customer_id, query):
            response = orchestrator(prompt)
        print(f"Response: {response}")
        print_trace_hint()
    flush_logs()


def run_chat() -> None:
    """Interactive terminal chat - educational mode."""
    W = _C.W

    # ── Welcome banner ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
    print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

    # ── Test customers ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.BOLD}Test Customers{_C.RESET}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
    print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
    for cid, name, tier, order, product in TEST_CUSTOMERS:
        tier_col = _C.INV if tier == 'Premium' else _C.GRY
        print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
              f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print()

    customer_id = (
        input(f"  Enter Customer ID (default: CUST-001): ").strip()
        or "CUST-001"
    )
    session_id  = str(uuid.uuid4())[:8]
    print()
    print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
    print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
    print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
    print()

    # ── Build agents and show initialization order.
    print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph(verbose=True)
    print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
    print()

    # ── Conversation loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input(
                f"  {_C.BOLD}You >{_C.RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print(f"  {_C.GRY}Session ended.{_C.RESET}")
            break

        prompt  = (f"[Session ID: {session_id}] "
                   f"[Customer ID: {customer_id}] {user_input}")
        t0_turn = time.time()

        # ── Install proxy, run orchestrator (traced), restore stdout ───
        trace.new_turn()
        sys.stdout = _trace_writer
        try:
            with tracer.trace_request(session_id, customer_id, user_input):
                response = orchestrator(prompt)
        finally:
            sys.stdout = _real_stdout   # always restore, even on exception

        elapsed = time.time() - t0_turn

        # ── Resolve the final customer-facing text ────────────────────
        final_state = _read_workflow_state(session_id) or {}
        comm_result = final_state.get('communication_agent', '')
        text = _strip_xml_tags(comm_result or str(response))

        # ── DynamoDB workflow state summary ───────────────────────────
        trace.summary(session_id, elapsed)

        # ── Final customer-facing response ────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        for line in text.splitlines():
            print(f"  {line}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        if tracer.last_trace_id:
            print(f"  {_C.GRY}X-Ray trace : {tracer.last_trace_id}"
                  f"{'' if tracer.last_published else '  (not published)'}{_C.RESET}")
        print()
    flush_logs()


def run_invoke(message: str, customer_id: str = "CUST-001") -> None:
    """Send one message to the deployed AgentCore Runtime and print the reply."""
    session_id = str(uuid.uuid4())[:8]
    print(f"Invoking {config.AGENTCORE_RUNTIME_ARN}")
    print(f"Session: {session_id} | Customer: {customer_id}")
    print(f"Query: {message}\n")
    result = invoke_agent(session_id, customer_id, message)
    print(f"Response: {result.get('result', result)}")
    if result.get('trace_id'):
        print(f"X-Ray trace: {result['trace_id']}")


def run_serve() -> None:
    """
    HTTP entry point executed inside Amazon Bedrock AgentCore Runtime.

    BedrockAgentCoreApp (bedrock-agentcore SDK) exposes the contract the
    runtime expects - POST /invocations and GET /ping on port 8080 - and hands
    each request payload to the function decorated with @app.entrypoint.

    Request payload (see invoke_agent):
        {"prompt": "<customer message>", "customer_id": "CUST-001", "session_id": "abc12345"}
    Response:
        {"result": "<final customer-facing text>", "session_id": ..., "trace_id": ...}

    The five-agent graph is built once (first request) and reused. Guardrail,
    tracing and logging are applied exactly as in the local test/chat modes,
    from the runtime's environment variables.
    """
    from bedrock_agentcore import BedrockAgentCoreApp

    os.environ.setdefault('AGENT_RUNTIME_MODE', 'agentcore-runtime')
    if os.environ.get('AGENT_LOG_GROUP') and 'AGENT_LOG_TO_CLOUDWATCH' not in os.environ:
        os.environ['AGENT_LOG_TO_CLOUDWATCH'] = 'true'

    app   = BedrockAgentCoreApp()
    lock  = threading.Lock()
    graph = {}

    def _orchestrator():
        with lock:
            if 'agent' not in graph:
                setup_logging()
                graph['agent'] = build_agent_graph()
        return graph['agent']

    @app.entrypoint
    def invoke(payload, context=None):
        payload     = payload or {}
        prompt      = payload.get('prompt') or payload.get('message') or ''
        customer_id = payload.get('customer_id') or 'CUST-001'
        session_id  = payload.get('session_id') or (
            getattr(context, 'session_id', None) or uuid.uuid4().hex)[:8]
        if not prompt:
            return {'error': "payload must include 'prompt'"}

        enriched = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {prompt}"
        with tracer.trace_request(session_id, customer_id, prompt):
            response = _orchestrator()(enriched)

        state = _read_workflow_state(session_id) or {}
        text  = _strip_xml_tags(state.get('communication_agent', '') or str(response))
        flush_logs()
        return {'result': text, 'session_id': session_id, 'customer_id': customer_id,
                'trace_id': tracer.last_trace_id}

    app.run()


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''

    # Inside the AgentCore Runtime package (marker file next to this script)
    # the entry point is started without arguments -> serve HTTP.
    if not command and os.path.exists(os.path.join(_SRC_DIR, _RUNTIME_MARKER)):
        command = 'serve'

    if command == 'deploy':
        deploy_all()

    elif command == 'serve':
        run_serve()

    elif command == 'test':
        run_test_scenarios()

    elif command == 'chat':
        run_chat()

    elif command == 'invoke':
        if len(sys.argv) < 3:
            print('Usage: python src/agent_orchestrator.py invoke "<message>" [CUSTOMER_ID]')
            sys.exit(1)
        run_invoke(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "CUST-001")

    else:
        print("Usage:")
        print("  python src/agent_orchestrator.py deploy           # Deploy to AgentCore (Tasks 3-6)")
        print("  python src/agent_orchestrator.py test             # Run the 3 test scenarios locally")
        print("  python src/agent_orchestrator.py chat             # Interactive terminal chat")
        print("  python src/agent_orchestrator.py invoke \"<msg>\"   # Call the deployed runtime")
        print("  python src/agent_orchestrator.py serve            # HTTP server (used inside AgentCore Runtime)")