"""Demonstração sintética de atualização de conhecimento conversacional.

Compara um classificador de respostas estático, ajuste com dados recentes,
ajuste com repetição de exemplos antigos e busca em documentos versionados.
Não é uma avaliação de LLM nem de um mecanismo completo de RAG.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import string
from copy import deepcopy

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier


DEFAULT_SEED = 31
ANSWERS = ("aplicativo", "balcao", "site", "telefone")
TRAIN_TEMPLATES = (
    "Qual é o canal do serviço {name}?",
    "Como acessar o serviço {name}?",
    "Por onde faço o pedido do serviço {name}?",
    "Onde encontro atendimento para {name}?",
    "Quero usar {name}; qual canal devo procurar?",
    "Informe o meio de acesso a {name}.",
)
TEST_TEMPLATES = (
    "Em que canal posso solicitar {name}?",
    "Preciso de {name}. Onde sou atendido?",
    "Qual canal atende pedidos de {name}?",
)


def make_facts(seed: int) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    rng = random.Random(seed)
    names: list[str] = []
    while len(names) < 40:
        candidate = "".join(rng.choices(string.ascii_lowercase, k=9))
        if candidate not in names:
            names.append(candidate)

    groups = {
        "estavel": names[:20],
        "alterado": names[20:30],
        "novo": names[30:40],
    }
    old = {name: ANSWERS[i % 4] for i, name in enumerate(names[:30])}
    current = old.copy()
    for name in groups["alterado"]:
        current[name] = ANSWERS[(ANSWERS.index(old[name]) + 1) % 4]
    for i, name in enumerate(groups["novo"]):
        current[name] = ANSWERS[(i + 2) % 4]
    return old, current, groups


def examples(facts: dict[str, str], templates: tuple[str, ...]) -> list[tuple[str, str]]:
    return [
        (template.format(name=name), answer)
        for name, answer in facts.items()
        for template in templates
    ]


def learn(
    model: SGDClassifier,
    vectorizer: HashingVectorizer,
    records: list[tuple[str, str]],
    epochs: int,
    rng: random.Random,
    first_update: bool = False,
) -> None:
    rows = records.copy()
    first_batch = first_update
    for _ in range(epochs):
        rng.shuffle(rows)
        inputs = vectorizer.transform([question for question, _ in rows])
        labels = [answer for _, answer in rows]
        if first_batch:
            model.partial_fit(inputs, labels, classes=ANSWERS)
            first_batch = False
        else:
            model.partial_fit(inputs, labels)


def evaluate_model(
    model: SGDClassifier,
    vectorizer: HashingVectorizer,
    current: dict[str, str],
    groups: dict[str, list[str]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for group, names in groups.items():
        questions = [template.format(name=name) for name in names for template in TEST_TEMPLATES]
        expected = [current[name] for name in names for _ in TEST_TEMPLATES]
        predicted = model.predict(vectorizer.transform(questions))
        result[group] = round(sum(p == y for p, y in zip(predicted, expected)) / len(expected), 3)
    return result


def evaluate_lookup(current: dict[str, str], groups: dict[str, list[str]]) -> dict[str, float]:
    # Documentos sintéticos usam apenas a versão vigente de cada fato.
    names = list(current)
    documents = [f"O canal de {name} é {current[name]}." for name in names]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    document_vectors = vectorizer.fit_transform(documents)
    result: dict[str, float] = {}
    for group, members in groups.items():
        questions = [template.format(name=name) for name in members for template in TEST_TEMPLATES]
        expected = [current[name] for name in members for _ in TEST_TEMPLATES]
        scores = (vectorizer.transform(questions) @ document_vectors.T).toarray()
        predicted = [current[names[index]] for index in scores.argmax(axis=1)]
        result[group] = round(sum(p == y for p, y in zip(predicted, expected)) / len(expected), 3)
    return result


def run(n_features: int = 512, seed: int = DEFAULT_SEED) -> dict[str, object]:
    old, current, groups = make_facts(seed)
    vectorizer = HashingVectorizer(
        analyzer="char", ngram_range=(3, 5), n_features=n_features, alternate_sign=False
    )
    baseline = SGDClassifier(
        loss="log_loss", alpha=0.00001, learning_rate="constant", eta0=0.1, random_state=seed
    )
    learn(baseline, vectorizer, examples(old, TRAIN_TEMPLATES), 40, random.Random(seed), True)

    recent = {name: current[name] for group in ("alterado", "novo") for name in groups[group]}
    recent_rows = examples(recent, TRAIN_TEMPLATES)
    updated = deepcopy(baseline)
    learn(updated, vectorizer, recent_rows, 30, random.Random(seed + 1))

    historical = {name: current[name] for name in groups["estavel"]}
    replay_rows = recent_rows + examples(historical, TRAIN_TEMPLATES[:2])
    replay = deepcopy(baseline)
    learn(replay, vectorizer, replay_rows, 30, random.Random(seed + 1))

    return {
        "seed": seed,
        "n_features": n_features,
        "facts": {group: len(names) for group, names in groups.items()},
        "questions_per_group": {group: len(names) * len(TEST_TEMPLATES) for group, names in groups.items()},
        "accuracy": {
            "static": evaluate_model(baseline, vectorizer, current, groups),
            "recent_only": evaluate_model(updated, vectorizer, current, groups),
            "recent_with_replay": evaluate_model(replay, vectorizer, current, groups),
            "versioned_lookup": evaluate_lookup(current, groups),
        },
    }


def save_plot(result: dict[str, object], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = result.get("mean_accuracy", result.get("accuracy", {}))
    methods = ("static", "recent_only", "recent_with_replay", "versioned_lookup")
    labels = ("Estático", "Recente", "Replay", "Busca")
    colors = ("#a64942", "#d18b32", "#4385a7", "#399565")
    groups = (("estavel", "Conhecimento estável"), ("alterado", "Conhecimento alterado"),
              ("novo", "Conhecimento novo"))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (group, title) in zip(axes, groups):
        values = [scores[method][group] for method in methods]
        bars = ax.bar(range(4), values, color=colors, width=0.7)
        ax.set_title(title)
        ax.set_ylim(0, 1.08)
        ax.set_xticks(range(4), labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Acurácia")
    figure.suptitle("Atualização de fatos: comparação em dados sintéticos", fontsize=14)
    figure.text(0.5, 0.01, "Média de 5 sementes; 512 atributos. Busca usa apenas documentos atuais.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Imprime resultado em JSON")
    parser.add_argument("--features", type=int, default=512, help="Tamanho do espaço de atributos")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semente da simulação")
    parser.add_argument("--seeds", type=int, nargs="+", help="Calcula médias de várias sementes")
    parser.add_argument("--plot", help="Salva gráfico em SVG ou PNG")
    args = parser.parse_args()
    if args.seeds:
        trials = [run(args.features, seed) for seed in args.seeds]
        result = {
            "seeds": args.seeds,
            "n_features": args.features,
            "facts_per_trial": trials[0]["facts"],
            "mean_accuracy": {
                method: {
                    group: round(statistics.mean(trial["accuracy"][method][group] for trial in trials), 3)
                    for group in ("estavel", "alterado", "novo")
                }
                for method in ("static", "recent_only", "recent_with_replay", "versioned_lookup")
            },
        }
    else:
        result = run(args.features, args.seed)
    if args.plot:
        save_plot(result, args.plot)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Precisão por grupo (dados sintéticos):")
        for method, values in result.get("mean_accuracy", result.get("accuracy", {})).items():
            print(f"{method:20s} " + " ".join(f"{group}={score:.3f}" for group, score in values.items()))
