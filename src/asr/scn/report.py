"""Update measured audit sections while preserving their explanatory Markdown."""

import json

from common.errors import ResourceError
from common.settings import SCN_EVIDENCE

BEGIN = "<!-- BEGIN GENERATED -->"
END = "<!-- END GENERATED -->"


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
        "",
    ]


def write_section(path, lines):
    document = path.read_text(encoding="utf-8")
    if document.count(BEGIN) != 1 or document.count(END) != 1:
        raise ResourceError(f"Expected one generated section: {path}")
    before, remaining = document.split(BEGIN)
    _, after = remaining.split(END)
    path.write_text(
        before + BEGIN + "\n\n" + "\n".join(lines) + "\n" + END + after,
        encoding="utf-8",
    )


def audit_tables(catalog, stats):
    total = stats["total"]
    lines = [
        f"Source evidence checked: **{catalog['checked_on']}**. "
        f"{len(catalog['sources']):,} catalog entries; "
        f"{total['records']:,} screened units / {total['tokens']:,} tokens; "
        f"{total['reviewed_records']:,} context-reviewed units / {total['reviewed_tokens']:,} tokens.",
        "",
        "## Measured compatible material",
        "",
    ]
    lines += table(
        [
            "Category",
            "Raw tokens",
            "Exact dedup",
            "Prose",
            "UI",
            "Lexical",
            "Reviewed units",
        ],
        (
            [
                category,
                *(
                    f"{row[key]:,}"
                    for key in (
                        "tokens",
                        "deduplicated_tokens",
                        "prose_tokens",
                        "localization_tokens",
                        "lexical_tokens",
                        "reviewed_records",
                    )
                ),
            ]
            for category, row in stats["categories"].items()
        ),
    )
    current = stats["current_cohort"]["total"]
    text = stats["text_categories"]["verified_conforming"]
    lines += [
        f"Verified text excluding lexical entries: **{text['tokens']:,} raw / {text['deduplicated_tokens']:,} deduplicated tokens**. "
        f"Current cohort excluding the older Wikipedia snapshot and documented RDF reproductions: **{current['tokens']:,} raw / {current['deduplicated_tokens']:,} deduplicated tokens**.",
        "",
        "## Rights intersection",
        "",
    ]
    lines += table(
        [
            "Category",
            "Conditional-inclusive training tokens",
            "Exact dedup",
            "Unconditionally permitted tokens",
        ],
        (
            [
                category,
                f"{row['tokens']:,}",
                f"{row['deduplicated_tokens']:,}",
                f"{stats['unconditional_training_intersection'][category]['tokens']:,}",
            ]
            for category, row in stats["research_training_intersection"].items()
        ),
    )
    lines += ["## Current Wikipedia", ""]
    lines += table(
        ["Article category", "Articles"],
        stats["wikipedia"]["article_categories"].items(),
    )
    mixed = stats["wikipedia"]["verified_passages_in_mixed_articles"]
    lines += [
        f"Verified passages within mixed articles: {mixed['records']:,} units / {mixed['tokens']:,} tokens.",
        "",
        "## LiITA measurements",
        "",
    ]
    lines += table(
        [
            "Source",
            "Units",
            "Lemmas",
            "Forms",
            "Variants",
            "Lexical tokens",
            "UD overlap units",
            "Novel dedup tokens",
        ],
        (
            [
                name,
                *(
                    f"{entry['measured'][key]:,}"
                    for key in (
                        "records",
                        "lemmas",
                        "lexical_forms",
                        "variant_forms",
                        "lexical_tokens",
                        "treebank_overlap_records",
                        "deduplicated_novel_tokens",
                    )
                ),
            ]
            for name, entry in stats["sources"].items()
            if name.startswith("scn_liita_")
        ),
    )
    return lines


def inventory_rows(catalog, stats, access):
    lines = [
        f"Evidence review: **{catalog['checked_on']}**. {len(catalog['sources']):,} entries, including all {len(catalog['original_source_ids'])} original entries.",
        "",
    ]
    lines += table(
        [
            "Source",
            "Acquisition",
            "Collection",
            "Analysis / training",
            "DOS24 inference",
            "Raw / dedup tokens",
            "Reviewed units",
        ],
        (
            [
                f"[{name}]({source['source_url']})",
                source["acquisition_status"],
                source["audit"]["bulk_collection"]["value"],
                source["audit"]["research_and_training"]["local_analysis"]["value"]
                + " / "
                + source["audit"]["research_and_training"]["model_training"]["value"],
                source["audit"]["dos24"]["value"],
                f"{stats['sources'][name]['measured']['tokens']:,} / {stats['sources'][name]['measured']['deduplicated_tokens']:,}"
                if stats["sources"][name]["measured"]["records"]
                else "not measured",
                stats["sources"][name]["measured"]["reviewed_records"],
            ]
            for name, source in catalog["sources"].items()
        ),
    )
    lines += ["## Source findings", ""]
    for name, source in catalog["sources"].items():
        observation = access.get(name, {})
        lines.append(
            f"- **{name}** — {source['name']}. {source['reason']} "
            f"Access: {observation.get('checked_on', 'unresolved')}, {observation.get('outcome', 'unresolved')}. "
            f"{observation.get('explanation', '')} "
            f"[Source]({source['source_url']}); [license/scope]({source['license_url']})."
        )
    return lines


def write_report(root, catalog, stats):
    access = json.loads(
        (root / SCN_EVIDENCE / "sources/access-audit.json").read_text()
    )["sources"]
    directory = root / "docs/scn/audits"
    write_section(directory / "dos24.md", audit_tables(catalog, stats))
    write_section(directory / "inventory.md", inventory_rows(catalog, stats, access))
    print("Updated docs/scn/audits/{dos24,inventory}.md")
