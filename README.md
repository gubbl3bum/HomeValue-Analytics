# HomeValue-Analytics

Analitics tool for housing prices from given CSV. CSV must have correct attributes to get app working.

## Reqiurements

* Runs only on python 3.12 <=
* Rest is in `requirements.txt`

## Setup scripts

Every script reqiures to be ran from `.venv`:

```shell
python3 -m venv .venv

source ./.venv/Scripts/activate # activate venv (Linux)
call venv\Scripts\activate      # activate venv (Windows)
```

Dev (setting developer environment)

Needs to have activated `.venv`.

```shell
pip install requirements.txt # download dependencies

streamlit run src/main.py # run in browser
python src/webview_app.py # run desktop view
streamlit run src/main.py --server.enableStaticServing true # debug streamlit
```

Build (package to desktop .`exe`)

```shell
cd / # go to project root

pyinstaller webview_app.spec
```

Prod (Running packaged app)

```shell
cd dist/

# run executable
```
