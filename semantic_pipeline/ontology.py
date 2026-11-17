"""Shared node vocabulary: categories and name types.

Two levels:

* Categories: one set for every node kind. These are AutoMap's meta-network
  categories, with group folded into agent, and without action, since no verbs
  are extracted. The meta-network's cells are pairs of categories.
* Name types: the fine types the NER step gives to names. Each maps to one
  category.

Names (node_type "entity") are specific nodes; common nouns (node_type
"concept") are generic nodes. AutoMap's specific/generic distinction is
therefore carried by node_type.

Layer 2 records the name type; Layer 3 maps it to its category (ontology mapping
belongs to Layer 3), so changing the mapping never requires re-running
extraction. Common nouns get their categories from the generated category table
(concept_table.py), which Layer 3 applies. Everything here is domain-general
and language-agnostic: the definitions are in English and nothing depends on
one language's spelling.
"""

from __future__ import annotations

from dataclasses import dataclass

ENTITY = "entity"
CONCEPT = "concept"
UNASSIGNED = "unassigned"

CATEGORIES: dict[str, str] = {
    "agent": "People and groups of people who can act.",
    "role": "A position or function an agent holds.",
    "organization": "A body with members or a formal structure.",
    "location": "A geographic or built place.",
    "event": "Something that happens at a time.",
    "task": "An activity people carry out.",
    "resource": "Something used, made, owned or consumed.",
    "knowledge": "Information, ideas, skills and topics, and works that carry them.",
    "attribute": "A property or state of an agent or thing.",
    "time": "A period or point in time.",
    UNASSIGNED: "No category assigned, or fits none.",
}

# Common-noun examples per category, in several languages and from several
# domains. Shown to models that assign categories.
CATEGORY_EXAMPLES: dict[str, str] = {
    "agent": "child, Nachbarin, fans, público",
    "role": "teacher, Trainerin, médico, moderator",
    "organization": "company, Verein, entreprise, team",
    "location": "city, Strand, cocina, station",
    "event": "concert, Unfall, boda, storm",
    "task": "training, Reparatur, compra, cleaning",
    "resource": "money, Auto, agua, medicine",
    "knowledge": "study, Rezept, idioma, mathematics",
    "attribute": "age, Gesundheit, courage, taille",
    "time": "summer, Woche, futuro, evening",
}

# The rule separating the two person categories. Shown to models that assign categories.
AGENT_ROLE_RULE = (
    "Nouns for occupations and functions are role; nouns that describe people in other "
    "ways (age, relation, residence, fandom) are agent."
)


@dataclass(frozen=True)
class NameType:
    """A fine type for names, with the definition shown to models and annotators."""

    name: str
    category: str
    definition: str
    includes: str
    excludes: str = ""


NAME_TYPES: tuple[NameType, ...] = (
    NameType(
        "Person", "agent",
        "A specific human being, real or fictional, referred to by name.",
        "full names, surnames, first names, nicknames, a person's social-media handle",
        "titles and roles",
    ),
    NameType(
        "Group", "agent",
        "People referred to collectively by a proper name or proper adjective, "
        "without formal membership.",
        "nationalities, ethnic and religious groups, named communities and fan bases, "
        "named generations",
        "collectives named by common nouns (residents, nurses)",
    ),
    NameType(
        "Organization", "organization",
        "A named body with members or a formal structure.",
        "companies, public authorities and institutions, associations, clubs and teams, "
        "schools and universities, hospitals as institutions, media outlets, bands, "
        "named committees",
        "the organization's products",
    ),
    NameType(
        "Place", "location",
        "A named geographic or built location.",
        "continents, countries, regions, cities, districts, streets and addresses, "
        "buildings and venues, rivers, mountains and other natural features",
    ),
    NameType(
        "Event", "event",
        "A named occurrence bound to a time.",
        "festivals, fairs and conferences, matches and tournaments, named storms and "
        "disasters, holidays, named meetings",
        "generic occurrences (a heat wave, the meeting)",
    ),
    NameType(
        "Work", "knowledge",
        "A named creative, intellectual or legal work.",
        "books, films, songs, shows and video series, artworks, articles and studies, "
        "websites and web addresses, laws, regulations and standards",
        "the organization behind it",
    ),
    NameType(
        "Product", "resource",
        "A named product, brand, model or service.",
        "devices, vehicles, apps and online platforms used as services, software, "
        "brand-name medicines, food and consumer brands",
        "the company acting as a company",
    ),
    NameType(
        "Other", UNASSIGNED,
        "A name that fits none of the types above.",
        "languages, awards, named projects and programs",
    ),
)

OTHER = "Other"
TYPE_NAMES: tuple[str, ...] = tuple(t.name for t in NAME_TYPES)

_CATEGORY_OF = {t.name: t.category for t in NAME_TYPES}
_BY_CASEFOLD = {t.name.casefold(): t.name for t in NAME_TYPES}


def name_type(value: str | None) -> str:
    """The name type matching `value`, ignoring case; OTHER if there is none."""
    return _BY_CASEFOLD.get((value or "").strip().casefold(), OTHER)


def category_of(type_name: str | None) -> str:
    """The category of a name type; UNASSIGNED for anything else."""
    return _CATEGORY_OF.get(type_name or "", UNASSIGNED)
