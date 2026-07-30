# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Markdown builder for sphinx-llm output locations."""

import os
from typing import Callable, Optional

from docutils import nodes
from docutils.io import StringOutput
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util.osutil import ensuredir, os_path
from sphinx_markdown_builder.builder import (
    MarkdownBuilder,
    get_mod_time_if_exists,
    io_handler,
)
from sphinx_markdown_builder.translator import MarkdownTranslator


def _standard_output_path(docname: str, suffix: str) -> str:
    return f"{docname}{suffix}"


def _html_file_suffix_output_path(docname: str, suffix: str) -> str:
    return f"{docname}.html{suffix}"


def _dirhtml_file_suffix_output_path(docname: str, suffix: str) -> str:
    if docname == "index" or docname.endswith("/index"):
        return f"{docname}.html{suffix}"
    return f"{docname}/index.html{suffix}"


def _dirhtml_url_suffix_output_path(docname: str, suffix: str) -> str:
    if docname == "index":
        return f"index{suffix}"
    if docname.endswith("/index"):
        return f"{docname.removesuffix('/index')}{suffix}"
    return f"{docname}{suffix}"


def _dirhtml_replace_output_path(docname: str, suffix: str) -> str:
    if docname == "index" or docname.endswith("/index"):
        return f"{docname}{suffix}"
    return f"{docname}/index{suffix}"


OUTPUT_PATH_BUILDERS: dict[str, Callable[[str, str], str]] = {
    "standard": _standard_output_path,
    "html-file-suffix": _html_file_suffix_output_path,
    "dirhtml-file-suffix": _dirhtml_file_suffix_output_path,
    "dirhtml-url-suffix": _dirhtml_url_suffix_output_path,
    "dirhtml-replace": _dirhtml_replace_output_path,
}


def output_path_for_docname(docname: str, layout: str, suffix: str = ".md") -> str:
    """Return the Markdown output path for a Sphinx document."""
    try:
        build_output_path = OUTPUT_PATH_BUILDERS[layout]
    except KeyError as exc:
        raise ValueError(f"Unknown Markdown output layout: {layout!r}") from exc
    return build_output_path(docname, suffix)


class SphinxLlmMarkdownTranslator(MarkdownTranslator):
    """Generate links for a page location or the output root."""

    def _adjust_url(self, url: str) -> str:
        if (
            not self.config.llms_txt_markdown_root_links
            or not self.config.markdown_http_base
        ):
            return super()._adjust_url(url)

        if not url:
            url = self.builder.get_target_uri(self.builder.current_doc_name)
        return f"{self.config.markdown_http_base.rstrip('/')}/{url}"

    def _fetch_ref_uri(self, node: nodes.reference) -> str:
        uri = super()._fetch_ref_uri(node)
        if (
            self.config.llms_txt_markdown_root_links
            and node.get("internal", self.status.default_ref_internal)
            and node.get("refid") is not None
        ):
            uri = self.builder.get_target_uri(self.builder.current_doc_name)
            if self.config.markdown_http_base:
                uri = f"{self.config.markdown_http_base.rstrip('/')}/{uri}"
            return f"{uri}#{node['refid']}"
        return uri


class SphinxLlmMarkdownBuilder(MarkdownBuilder):
    """Write Markdown at the locations published by sphinx-llm."""

    name = "llms-markdown"
    default_translator_class = SphinxLlmMarkdownTranslator

    def __init__(self, app: Sphinx, env: Optional[BuildEnvironment] = None) -> None:
        super().__init__(app, env)
        self.output_layout = "standard"

    def init(self) -> None:
        super().init()
        self.output_layout = self.config.llms_txt_markdown_layout
        if self.output_layout not in OUTPUT_PATH_BUILDERS:
            raise ValueError(f"Unknown Markdown output layout: {self.output_layout!r}")

    def output_path(self, docname: str) -> str:
        """Return the output path for a Sphinx document."""
        return output_path_for_docname(
            docname,
            self.output_layout,
            self.config.markdown_file_suffix,
        )

    def _get_target_mtime(self, doc_name: str):
        target_name = os.path.join(self.outdir, os_path(self.output_path(doc_name)))
        return get_mod_time_if_exists(target_name, log_error=False)

    def get_target_uri(self, docname: str, typ: Optional[str] = None) -> str:
        return self.output_path(docname)

    def get_relative_uri(self, from_: str, to: str, typ: Optional[str] = None) -> str:
        if self.config.llms_txt_markdown_root_links:
            return self.get_target_uri(to, typ)
        return super().get_relative_uri(from_, to, typ)

    def write_doc(self, docname: str, doctree: nodes.document) -> None:
        self.current_doc_name = docname
        self.sec_numbers = self.env.toc_secnumbers.get(docname, {})
        destination = StringOutput(encoding="utf-8")
        self.writer.write(doctree, destination)
        out_filename = os.path.join(self.outdir, os_path(self.output_path(docname)))
        ensuredir(os.path.dirname(out_filename))

        with io_handler(out_filename):
            with open(out_filename, "w", encoding="utf-8") as file:
                file.write(self.writer.output)
