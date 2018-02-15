from pathlib import Path

class AOrg():
    download_url = "http://archive.org/download/"
    files_templates = {
        "files": "{id}/{id}_files.xml",
        "meta": "{id}/{id}_meta.xml",
        "jp2zip": "{id}/{id}_jp2.zip",
        "jp2": "{id}/{id}_jp2/*.jp2",

        "jpgs_five": "{id}/{id}_jpg/{pagename}.jpg"
    }

    paths_templates = {
        "jp2" : "{id}/{id}_jp2/",
        "jpg": "{id}/{id}_jpg/",
        "files": "{id}/"
    }
    globs = {
        "jp2": "*.jp2",
        "jpg": "*.jpg"
    }

    def file(identifier, kind="files"):
        return Path(AOrg.files_templates[kind].format(id=identifier))

    def path(identifier, kind='files'):
        return Path(AOrg.paths_templates[kind].format(id=identifier))

    def url(identifier, kind='files'):
        return AOrg.download_url + str(AOrg.path(identifier, kind))

    def fname(identifier, kind='files'):
        return AOrg.path(identifier, kind).name

    def files(identifier, kind='jpg'):
        yield from AOrg.path(identifier,kind).rglob(AOrg.globs[kind])