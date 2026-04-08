uv run uvicorn server.app:app --host 0.0.0.0 --port 7860 &
SERVER_PID=$!
sleep 2
echo "Testing POST /reset with NO body:"
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:7860/reset
echo
echo "Testing POST /reset with EMPTY body:"
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
echo
kill $SERVER_PID 2>/dev/null
