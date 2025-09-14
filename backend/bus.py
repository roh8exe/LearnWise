# backend/bus.py
import asyncio
from collections import defaultdict

subscribers = defaultdict(list)

# backend/bus.py - add more debugging
async def publish(session_id, topic, msg):
    key = f"{session_id}:{topic}"
    print(f" PUBLISH {key}: {msg}")  # DEBUG
    queues = list(subscribers[key])
    print(f"   Delivering to {len(queues)} subscribers")
    for queue in queues:
        try:
            await queue.put(msg)
            print(f"   ✅ Delivered to queue {id(queue)}")
        except Exception as e:
            print(f"    Delivery failed: {e}")

async def subscribe(session_id, topic):
    key = f"{session_id}:{topic}"
    queue = asyncio.Queue()
    subscribers[key].append(queue)
    print(f"👂 SUBSCRIBED {key}, total={len(subscribers[key])}")
    try:
        while True:
            msg = await queue.get()
            print(f" SUBSCRIBE YIELD {key}: {msg}")  # DEBUG
            yield msg
    finally:
        subscribers[key].remove(queue)
        print(f" UNSUBSCRIBED {key}, remaining={len(subscribers[key])}")
