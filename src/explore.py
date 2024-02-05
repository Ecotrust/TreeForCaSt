""" Utility functions for browsing and downloading from the ForestPlots STAC"""
import os
import wget
from pystac import Item
from pathlib import Path


def has_datasets(item, collection_ids: list):
    """Check if the item has the datasets in the given collection_id list.

    Args:
        item (pystac.Item): Plot item to check.
        collection_ids (list): List like object with collection IDs to check for.
    """
    assert isinstance(collection_ids, list or tuple)

    # get the source absolute path to dataset items
    sources = [link.absolute_href for link in item.links if link.rel == "source"]
    # Load the dataset items from file. This is faster than using catalog.get_item()
    _items = [Item.from_file(source) for source in sources]

    return set(collection_ids).issubset([_item.collection_id for _item in _items])


def count_sources(item):
    """Count the number of sources for the given item."""
    # get the source absolute path to dataset items
    sources = [link.absolute_href for link in item.links if link.rel == "source"]
    return len(sources)


def get_item_assets(item):
    """ """
    # get the source absolute path to dataset items
    sources = [link.absolute_href for link in item.links if link.rel == "source"]
    # Load the dataset items from file. This is faster than crawling the catalog
    return [Item.from_file(source) for source in sources]


def download_assets(item, outpath):
    """Download all assets of the given item and update the hrefs."""
    item_path = Path(item.self_href).parent.stem
    new_item = item.clone()
    outpath = Path(outpath)
    for k, v in item.assets.items():
        parts = Path(v.href).parts[4:]
        if len(parts) > 2:
            collection, year, filename = Path(v.href).parts[4:]
            outfile = outpath / collection / year / item_path / filename
        else:
            collection, filename = parts
            outfile = outpath / collection / item_path / filename

        # Download the asset
        if not outfile.exists():
            outfile.parent.mkdir(parents=True, exist_ok=True)
            wget.download(v.href, out=outfile.as_posix())

        # Update asset href to point to local file
        new_item.assets[k].href = os.path.abspath(outfile)

    return new_item


def update_href(item, new_dir, new_collection):
    """Update the href of the given item."""
    self_href = os.path.join(
        new_dir, item.collection_id, item.id, Path(item.self_href).name
    )
    new_item = item.clone()
    new_item.set_self_href(self_href)
    new_item.set_parent(new_collection)

    return new_item
