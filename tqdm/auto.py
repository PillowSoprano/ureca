def tqdm(iterable=None, *args, **kwargs):
    """Lightweight tqdm stub for environments without the package installed."""

    if iterable is None:
        return _TqdmPlaceholder()
    return iterable


class _TqdmPlaceholder:
    def __call__(self, iterable, *_, **__):
        return iterable

