mkdir -p ~/.streamlit/
echo "[general]
email = \"your-email@example.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
