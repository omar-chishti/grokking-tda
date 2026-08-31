"""A derived ETbb small-capitals face: matplotlib cannot ask a font for a feature."""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm

CACHE = Path(__file__).parent / ".fontcache"
FAMILY = "ETbb SC"

_STYLES = {"Regular": "", "Italic": "-Italic", "Bold": "-Bold", "BoldItalic": "-BoldItalic"}


def _substitutions(font, tag: str) -> dict[str, str]:
    gsub = font["GSUB"].table
    out: dict[str, str] = {}
    for record in gsub.FeatureList.FeatureRecord:
        if record.FeatureTag != tag:
            continue
        for index in record.Feature.LookupListIndex:
            for table in gsub.LookupList.Lookup[index].SubTable:
                out.update(getattr(table, "mapping", {}))
    return out


def _derive(source: Path, target: Path, family: str, style: str, tags: tuple[str, ...]) -> None:
    from fontTools.ttLib import TTFont

    font = TTFont(source)
    mapping: dict[str, str] = {}
    for tag in tags:
        mapping.update(_substitutions(font, tag))

    glyphs = set(font.getGlyphOrder())
    for table in font["cmap"].tables:
        table.cmap = {
            code: mapping.get(name, name)
            if mapping.get(name, name) in glyphs
            else name
            for code, name in table.cmap.items()
        }

    full = f"{family} {style}" if style != "Regular" else family
    for record in font["name"].names:
        if record.nameID == 1:
            record.string = family
        elif record.nameID == 2:
            record.string = style
        elif record.nameID == 4:
            record.string = full
        elif record.nameID == 6:
            record.string = full.replace(" ", "")
        elif record.nameID in (16, 17):
            record.string = family if record.nameID == 16 else style
    font.save(target)


def install() -> str:
    CACHE.mkdir(exist_ok=True)
    for style, suffix in _STYLES.items():
        source = Path.home() / "Library" / "Fonts" / f"ETbb{suffix or '-Regular'}.otf"
        target = CACHE / f"ETbbSC{suffix or '-Regular'}.otf"
        if not source.exists():
            continue
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
            _derive(source, target, FAMILY, style, ("onum", "smcp"))
        fm.fontManager.addfont(str(target))
    return FAMILY
