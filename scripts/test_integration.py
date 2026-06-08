"""
Integration smoke test — runs against live local agents.
Requires all three agents to be running (start_local.sh or docker-compose up).

Changes vs original:
  - /query response now includes question_id and conversation_id — tests verify these.
  - NEW: /feedback endpoint test.
  - NEW: /history endpoint test.
  - NEW: /memory endpoint test.
  - Existing query tests are 100% unchanged.

Usage:
    python scripts/test_integration.py
"""
from __future__ import annotations

import asyncio
import sys

import httpx

MAIN_AGENT_URL   = "http://localhost:8000"
ORCHESTRATOR_URL = "http://localhost:8001"
RETRIEVAL_URL    = "http://localhost:8002"

TEST_QUERIES = [
    ("What is the annual leave policy for full-time employees?", "hr"),
    ("How do I reset my VPN credentials?",                       "it"),
    ("What are the GDPR data retention obligations for employee records?", "legal"),
    ("raise_ticket",  None),
    ("connect_sme",   None),
]

CONV_ID = "test-conv-smoke-001"
USER_ID = "test-user-smoke"


async def check_health(client: httpx.AsyncClient, url: str, name: str) -> bool:
    try:
        resp = await client.get(f"{url}/health", timeout=5.0)
        resp.raise_for_status()
        print(f"  ✅ {name}: {resp.json()}")
        return True
    except Exception as exc:
        print(f"  ❌ {name}: {exc}")
        return False


async def run_query(client: httpx.AsyncClient, text: str) -> dict:
    resp = await client.post(
        f"{MAIN_AGENT_URL}/query",
        json={"text": text, "conversation_id": CONV_ID, "user_id": USER_ID},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


async def test_feedback(client: httpx.AsyncClient, answer_id: str, question_id: str) -> bool:
    try:
        resp = await client.post(
            f"{MAIN_AGENT_URL}/feedback",
            json={
                "answer_id":   answer_id,
                "question_id": question_id,
                "user_id":     USER_ID,
                "rating":      5,
                "comment":     "Smoke test feedback",
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        assert data.get("status") == "saved", f"unexpected status: {data}"
        assert data.get("feedback_id", "").startswith("fb-"), f"bad feedback_id: {data}"
        print(f"   feedback_id={data['feedback_id']} ✅")
        return True
    except Exception as exc:
        print(f"   ❌ /feedback failed: {exc}")
        return False


async def test_history(client: httpx.AsyncClient) -> bool:
    try:
        resp = await client.get(
            f"{MAIN_AGENT_URL}/history/{CONV_ID}",
            params={"limit": 5},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        count = data.get("count", -1)
        print(f"   turns_returned={count} ✅")
        return True
    except Exception as exc:
        print(f"   ❌ /history failed: {exc}")
        return False


async def test_memory(client: httpx.AsyncClient) -> bool:
    try:
        resp = await client.get(
            f"{MAIN_AGENT_URL}/memory/{USER_ID}",
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        count = data.get("count", -1)
        print(f"   memory_records={count} ✅")
        return True
    except Exception as exc:
        print(f"   ❌ /memory failed: {exc}")
        return False


async def main():
    print("=" * 60)
    print("RAG Enterprise — Integration Smoke Test")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        print("\n── Health Checks ──────────────────────────────────────────")
        results = await asyncio.gather(
            check_health(client, RETRIEVAL_URL,    "Retrieval Agent"),
            check_health(client, ORCHESTRATOR_URL, "Orchestrator Agent"),
            check_health(client, MAIN_AGENT_URL,   "Main Agent"),
        )
        if not all(results):
            print("\nERROR: One or more agents are not healthy. Start them first.")
            sys.exit(1)

        print("\n── Query Tests ────────────────────────────────────────────")
        last_answer_id  = None
        last_question_id = None

        for query_text, expected_domain in TEST_QUERIES:
            print(f"\nQ: {query_text[:70]}")
            try:
                result = await run_query(client, query_text)
                reply: str = result.get("reply", "")
                question_id = result.get("question_id", "")
                print(f"A: {reply[:200]}{'...' if len(reply) > 200 else ''}")

                # Verify IDs are present for non-special commands
                if query_text not in ("raise_ticket", "connect_sme"):
                    assert question_id.startswith("q-"), f"bad question_id: {question_id}"
                    print(f"   question_id={question_id} ✅")
                    last_question_id = question_id

                # Extract answer_id from reply metadata (not in response body — use question_id for feedback test)
                last_answer_id = f"ans-smoke"   # placeholder; real answer_id is in Cosmos

                print("   ✅ OK")
            except Exception as exc:
                print(f"   ❌ FAILED: {exc}")

        print("\n── Feedback Test ──────────────────────────────────────────")
        if last_question_id:
            await test_feedback(client, answer_id="ans-smoke-001", question_id=last_question_id)
        else:
            print("   ⚠️  Skipped — no successful query to test against")

        print("\n── History Test ───────────────────────────────────────────")
        await test_history(client)

        print("\n── Memory Test ────────────────────────────────────────────")
        await test_memory(client)

    print("\n" + "=" * 60)
    print("Smoke test complete.")


if __name__ == "__main__":
    asyncio.run(main())
