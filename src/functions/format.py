from src.functions.labels import ABECEDARY, LOWER_ABECEDARY


VOID_STR = "\u2205"


def fmt_kparticion(k_partition) -> str:
    """Format a k-partition (tuple of (mechanism, alcance) frozenset pairs)."""
    labels = []
    for mech, alc in k_partition:
        mech_str = "".join(LOWER_ABECEDARY[i] for i in sorted(mech)) if mech else VOID_STR
        alc_str = "".join(ABECEDARY[i] for i in sorted(alc)) if alc else VOID_STR
        labels.append((mech_str, alc_str))

    widths = [max(len(m), len(a)) + 2 for m, a in labels]
    top = "".join(f"\u239b{a:^{w}}\u239e" for (_, a), w in zip(labels, widths))
    bot = "".join(f"\u239d{m:^{w}}\u23a0" for (m, _), w in zip(labels, widths))
    return f"{top}\n{bot}"


def fmt_biparticion_fuerza_bruta(
    parte_uno: list[tuple, tuple],
    parte_dos: list[tuple, tuple],
) -> str:
    mech_p, pur_p = parte_uno
    mech_d, purv_d = parte_dos

    purv_prim = "".join(ABECEDARY[j] for j in pur_p) if pur_p else VOID_STR
    mech_prim = "".join(LOWER_ABECEDARY[i] for i in mech_p) if mech_p else VOID_STR

    purv_dual = "".join(ABECEDARY[i] for i in purv_d) if purv_d else VOID_STR
    mech_dual = "".join(LOWER_ABECEDARY[j] for j in mech_d) if mech_d else VOID_STR

    width_prim = max(len(purv_prim), len(mech_prim)) + 2
    width_dual = max(len(purv_dual), len(mech_dual)) + 2

    return (
        f"\u239b{purv_prim:^{width_prim}}\u239e\u239b{purv_dual:^{width_dual}}\u239e\n"
        f"\u239d{mech_prim:^{width_prim}}\u23a0\u239d{mech_dual:^{width_dual}}\u23a0\n"
    )


def fmt_biparticion_q(
    prim: list[tuple[int, int]],
    dual: list[tuple[int, int]],
    to_sort: bool = True,
) -> str:
    top_prim, bottom_prim = _fmt_parte_q(prim, to_sort)
    top_dual, bottom_dual = _fmt_parte_q(dual, to_sort)
    return f"{top_prim}{top_dual}\n{bottom_prim}{bottom_dual}"


def _fmt_parte_q(parte: list[tuple[int, int]], to_sort: bool = True) -> tuple[str, str]:
    if to_sort:
        parte.sort(key=lambda x: x[1])

    purv, mech = [], []
    for time, idx in parte:
        purv.append(ABECEDARY[idx]) if time else mech.append(LOWER_ABECEDARY[idx])

    str_purv = "".join(purv) if purv else VOID_STR
    str_mech = "".join(mech) if mech else VOID_STR
    width = max(len(str_purv), len(str_mech)) + 2

    return f"\u239b{str_purv:^{width}}\u239e", f"\u239d{str_mech:^{width}}\u23a0"
