# HomeValue-Analytics

Analitics tool for housing prices from given CSV. CSV must have correct attributes to get app working.

## Reqiurements

* Runs only on python 3.12 <=
* Rest is in `requirements.txt`

## Setup scripts

### Create/Activate venv

Every script reqiures to be ran from `.venv`:

```shell
python3 -m venv .venv

source ./.venv/Scripts/activate # activate venv (Linux)
.\.venv\Scripts\Activate.ps1    # activate venv (Windows)

pip install -r requirements.txt # download dependencies
```

### Dev (setting developer environment)

Needs to have activated `.venv`.

```shell
streamlit run src/main.py # run in browser
python src/app.py # run desktop view
streamlit run src/main.py --server.enableStaticServing true # debug streamlit
```

### Build (package to desktop .`exe`)

```shell
cd ~/HomeValue-Analytics # go to project root

pyinstaller app.spec
```

### Prod (Running packaged app)

```shell
# build app

cd dist/

# run executable
```
