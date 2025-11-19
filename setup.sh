mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"664230033@webmail.npru.ac.th\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml