"""Dash chat interface (bonus) for the Île-de-France events RAG.

A thin demo client: it POSTs questions to the FastAPI ``/ask`` endpoint and renders the
grounded answer with its cited sources. The conversation history lives client-side in a
``dcc.Store`` (the API is stateless, so history is display-only and reset clears it). The
API must be running separately (``uvicorn src.api.main:app``).

    poetry run python interface/dash_app.py        # http://localhost:8050
    RAG_API_URL=http://localhost:8000 poetry run python interface/dash_app.py
"""

from __future__ import annotations

import os

import requests
from dash import Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


def ask_api(question: str, api_url: str = API_URL) -> dict:
    """POST a question to the RAG API; return its JSON or an ``{"error": ...}`` dict."""
    try:
        response = requests.post(
            f"{api_url}/ask", json={"question": question}, timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": f"API injoignable : {exc}"}


def _source_label(source: dict) -> str:
    """One-line label for a cited event: 'Titre — lieu, ville — dates'."""
    title = source.get("title") or "(sans titre)"
    location = ", ".join(str(source[k]) for k in ("venue", "city") if source.get(k))
    daterange = source.get("daterange") or ""
    return " — ".join(part for part in (title, location, daterange) if part)


def _render_sources(sources: list[dict]) -> html.Details:
    """Collapsible list of unique cited events (chunks of the same event are deduped)."""
    items, seen = [], set()
    for source in sources:
        key = source.get("url") or source.get("id") or source.get("title")
        if key in seen:
            continue
        seen.add(key)
        label = _source_label(source)
        url = source.get("url")
        items.append(
            html.Li(html.A(label, href=url, target="_blank") if url else label)
        )
    return html.Details(
        [html.Summary(f"Sources ({len(items)})"), html.Ul(items, className="sources-list")],
        className="sources",
    )


def render_message(entry: dict):
    """Render one history entry (user / assistant / error) into a chat bubble."""
    if entry.get("role") == "user":
        return html.Div(entry.get("content", ""), className="bubble user")

    if entry.get("error"):
        return html.Div(
            [html.B("Erreur : "), entry["error"]], className="bubble assistant error"
        )

    children = [html.Div(entry.get("content", ""), className="answer")]
    filters = entry.get("filters") or {}
    if filters:
        rendered = ", ".join(f"{k} = {v}" for k, v in filters.items())
        children.append(html.Div(f"Filtres détectés : {rendered}", className="filters"))
    sources = entry.get("sources") or []
    if sources:
        children.append(_render_sources(sources))
    return html.Div(children, className="bubble assistant")


app = Dash(__name__, title="RAG Événements Île-de-France")
server = app.server  # exposed for WSGI / containerized serving

app.layout = html.Div(
    className="app",
    children=[
        html.H1("Assistant — Événements culturels en Île-de-France"),
        html.P(
            "Posez une question en langage naturel ; les réponses sont fondées sur les "
            "événements indexés.",
            className="subtitle",
        ),
        dcc.Store(id="history", data=[]),
        dcc.Loading(
            html.Div(id="chat-window", className="chat-window"), type="dot"
        ),
        html.Div(
            className="input-row",
            children=[
                dcc.Input(
                    id="question",
                    type="text",
                    placeholder="Ex. : Quels concerts à Paris ce week-end ?",
                    className="question-input",
                    n_submit=0,
                ),
                html.Button("Envoyer", id="send", n_clicks=0, className="btn btn-send"),
                html.Button(
                    "Réinitialiser", id="reset", n_clicks=0, className="btn btn-reset"
                ),
            ],
        ),
    ],
)


@callback(
    Output("history", "data"),
    Output("question", "value"),
    Input("send", "n_clicks"),
    Input("question", "n_submit"),
    State("question", "value"),
    State("history", "data"),
    prevent_initial_call=True,
)
def submit(_n_clicks, _n_submit, question, history):
    """Send the question to the API and append the exchange to the history store."""
    if not question or not question.strip():
        raise PreventUpdate
    question = question.strip()
    history = (history or []) + [{"role": "user", "content": question}]

    result = ask_api(question)
    if "error" in result:
        history.append({"role": "assistant", "error": result["error"]})
    else:
        history.append(
            {
                "role": "assistant",
                "content": result.get("answer", ""),
                "sources": result.get("sources", []),
                "filters": result.get("filters", {}),
            }
        )
    return history, ""


@callback(Output("chat-window", "children"), Input("history", "data"))
def render(history):
    return [render_message(entry) for entry in (history or [])]


@callback(
    Output("history", "data", allow_duplicate=True),
    Input("reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset(_n_clicks):
    return []


if __name__ == "__main__":
    app.run(debug=True, port=8050)
