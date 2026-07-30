# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Sphinx extension to generate markdown files alongside HTML files.

This extension hooks into the Sphinx build process to create markdown versions
of all documents using the sphinx_markdown_builder.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path
from typing import Any, Optional, Union

import docutils.nodes
from sphinx.application import Sphinx
from sphinx.errors import ExtensionError
from sphinx.util import logging

from .markdown_builder import (
    SphinxLlmMarkdownBuilder,
    output_path_for_docname,
)
from .version import __version__

logger = logging.getLogger(__name__)


@dataclass
class MarkdownBuild:
    """Describe one Markdown subprocess build."""

    output_dir: Path
    layout: str
    primary: bool = False
    aggregate: bool = False
    process: Any = None
    log_path: Optional[Path] = None


class MarkdownGenerator:
    """Generates markdown files using sphinx_markdown_builder."""

    def __init__(self, app: Sphinx):
        self.app = app
        self.generated_markdown_files = []  # Track generated markdown files
        self._docname_by_output_file: dict[Path, str] = {}  # output file → docname
        self.outdir = None
        self.md_build_dir = None
        self.markdown_builds: list[MarkdownBuild] = []
        self.parallel = None

    def setup(self):
        """Set up the extension."""
        self.app.connect("builder-inited", self.build_llms_txt)

    def build_llms_txt(self, app: Sphinx):
        """Generate markdown files using sphinx_markdown_builder and concatenate them into llms.txt."""
        if not getattr(self.app.config, "llms_txt_enabled", True):
            logger.info(
                "llms.txt generation is disabled (llms_txt_enabled=False), skipping..."
            )
            return

        self.outdir = Path(app.builder.outdir)
        self.md_build_dir = self.outdir / "_markdown_build"
        self.parallel = getattr(self.app.config, "llms_txt_build_parallel", True)
        self.suffix_mode = getattr(self.app.config, "llms_txt_suffix_mode", "auto")

        # Backward compatibility: treat "both" as "auto"
        if self.suffix_mode == "both":
            self.suffix_mode = "auto"

        # Validate suffix_mode configuration
        valid_modes = {"file-suffix", "url-suffix", "auto", "replace"}
        if self.suffix_mode not in valid_modes:
            raise ExtensionError(
                f"Invalid llms_txt_suffix_mode: {self.suffix_mode!r}. "
                f"Must be one of {valid_modes}"
            )

        if app.builder and app.builder.name == "markdown":
            return

        if not app.builder or app.builder.name not in ["html", "dirhtml"]:
            logger.info(
                "llms.txt generation only works with HTML builders (html or dirhtml), skipping..."
            )
            return

        # Start the markdown builder subproces in the background
        if self.parallel:
            self.build_markdown_files()
        else:
            logger.info(
                "Option llms_txt_build_parallel is set to False, will build markdown files after the primary build is finished"
            )
            self.app.connect("build-finished", self.build_markdown_files, priority=100)
        # Once the primary build is finished, combine the markdown files
        self.app.connect("build-finished", self.combine_builds, priority=101)

    def combine_builds(self, app: Sphinx, exception: Union[Exception, None]):
        """Combine the markdown files into llms-full.txt and llms.txt and merge the build outputs together."""
        if exception:
            logger.warning("Skipping build combination due to build error")
            return

        if not self.markdown_builds:
            logger.warning(
                "Markdown build processes not found, skipping build output combination"
            )
            return

        for build in self.markdown_builds:
            if build.process is None:
                logger.error("Markdown build subprocess did not start")
                return
            if build.process.poll() is None:
                logger.info("Waiting for markdown build subprocess to finish...")
                build.process.wait()
                logger.info("Markdown build subprocess finished")
            if build.process.returncode != 0:
                logger.error(
                    f"Markdown build subprocess failed with return code {build.process.returncode}"
                )
                if build.log_path:
                    logger.error(build.log_path.read_text(encoding="utf-8"))
                return

        try:
            # Copy markdown files to the main output directory
            self.copy_markdown_files()

            # Concatenate all markdown files into llms-full.txt
            if getattr(self.app.config, "llms_txt_full_build", True):
                self.build_llms_full_txt()

            # Create sitemap in llms.txt
            self.create_sitemap()
        finally:
            # Clean up temporary build directory
            if self.md_build_dir.exists():
                shutil.rmtree(self.md_build_dir)

    def _page_layouts(self) -> list[str]:
        if self.app.builder.name == "html":
            if self.suffix_mode == "replace":
                return ["standard"]
            return ["html-file-suffix"]
        if self.suffix_mode == "file-suffix":
            return ["dirhtml-file-suffix"]
        if self.suffix_mode == "url-suffix":
            return ["dirhtml-url-suffix"]
        if self.suffix_mode == "replace":
            return ["dirhtml-replace"]
        return ["dirhtml-file-suffix", "dirhtml-url-suffix"]

    def _create_markdown_builds(self) -> list[MarkdownBuild]:
        builds = [
            MarkdownBuild(
                output_dir=self.md_build_dir / f"pages-{index}",
                layout=layout,
                primary=index == 0,
            )
            for index, layout in enumerate(self._page_layouts())
        ]
        if getattr(self.app.config, "llms_txt_full_build", True):
            builds.append(
                MarkdownBuild(
                    output_dir=self.md_build_dir / "aggregate",
                    layout=builds[0].layout,
                    aggregate=True,
                )
            )
        return builds

    def _markdown_build_command(self, build: MarkdownBuild) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            SphinxLlmMarkdownBuilder.name,
            "-t",
            "sphinx_llm_markdown",
            "-D",
            f"llms_txt_markdown_layout={build.layout}",
        ]
        if build.aggregate:
            command.extend(["-D", "llms_txt_markdown_root_links=1"])
        command.extend([str(self.app.srcdir), str(build.output_dir)])
        return command

    def build_markdown_files(self, *_):
        """Start the Markdown subprocess builds."""
        self.md_build_dir.mkdir(exist_ok=True)
        self.markdown_builds = self._create_markdown_builds()

        for build in self.markdown_builds:
            build.output_dir.mkdir(parents=True, exist_ok=True)
            command = self._markdown_build_command(build)
            logger.info(
                f"Spawning additional sphinx subprocess to build markdown files for llms.txt: {' '.join(command)}"
            )
            logfile = tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                prefix="sphinx_llm_output_",
                suffix=".log",
            )
            build.log_path = Path(logfile.name)
            logger.info(f"Subprocess output available at: {build.log_path}")
            try:
                build.process = subprocess.Popen(
                    command,
                    stdout=logfile,
                    stderr=logfile,
                )
            except Exception as exc:
                logger.error(f"Failed to run sphinx-build subprocess: {exc}")
            finally:
                logfile.close()

    def copy_markdown_files(self):
        """Copy Markdown files to the main output directory."""
        self.generated_markdown_files = []
        self._docname_by_output_file = {}
        page_builds = [build for build in self.markdown_builds if not build.aggregate]

        for build in page_builds:
            for docname in self.app.env.found_docs:
                relative_path = Path(output_path_for_docname(docname, build.layout))
                source_file = build.output_dir / relative_path
                if not source_file.exists():
                    continue
                target_file = self.outdir / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target_file)
                if build.primary:
                    self.generated_markdown_files.append(target_file)
                    self._docname_by_output_file[target_file] = docname

        logger.info(f"Generated {len(self.generated_markdown_files)} context files")

    def build_llms_full_txt(self):
        """Build the combined Markdown context file."""
        aggregate_build = next(
            build for build in self.markdown_builds if build.aggregate
        )
        llms_txt_path = self.outdir / "llms-full.txt"
        with open(llms_txt_path, "w", encoding="utf-8") as llms_txt:
            sorted_files = sorted(
                self.generated_markdown_files,
                key=lambda path: (
                    path.relative_to(self.outdir).as_posix()
                    not in {"index.html.md", "index.md"},
                    path.relative_to(self.outdir).as_posix(),
                ),
            )

            for md_file in sorted_files:
                docname = self._docname_by_output_file[md_file]
                aggregate_file = aggregate_build.output_dir / output_path_for_docname(
                    docname, aggregate_build.layout
                )
                with open(aggregate_file, encoding="utf-8") as f:
                    relative_path = md_file.relative_to(self.outdir).as_posix()
                    llms_txt.write(f"# {relative_path}\n\n")
                    llms_txt.write(f.read())
                    llms_txt.write("\n\n")
        logger.info(f"Concatenated full context into: {llms_txt_path}")

    def get_project_description(self) -> str:
        """Get the description of the project."""
        project_title = getattr(self.app.config, "project", "Documentation")
        if (
            hasattr(self.app.config, "llms_txt_description")
            and self.app.config.llms_txt_description
        ):
            return self.app.config.llms_txt_description

        try:
            meta_description = metadata(project_title).get("Description")
            if meta_description:
                return meta_description
        except PackageNotFoundError:
            pass

        if hasattr(self.app.config, "html_title") and self.app.config.html_title:
            return self.app.config.html_title

        return f"Documentation for {project_title}"

    def create_sitemap(self):
        """Create a markdown sitemap in llms.txt."""
        llms_txt_path = self.outdir / "llms.txt"

        with open(llms_txt_path, "w", encoding="utf-8") as sitemap:
            # Write the title
            project_title = getattr(self.app.config, "project", "Documentation")
            sitemap.write(f"# {project_title}\n\n")

            # Add description
            for line in self.get_project_description().strip().split("\n"):
                sitemap.write(f"> {line}\n")
            sitemap.write("\n\n")

            # Add project details if available
            if hasattr(self.app.config, "copyright") and self.app.config.copyright:
                sitemap.write(f"{self.app.config.copyright}\n\n")

            # Write the main content section
            sitemap.write("## Pages\n\n")

            # Sort files to ensure index.html.md comes first
            sorted_files = sorted(
                self.generated_markdown_files,
                key=lambda x: (x.name not in ("index.html.md", "index.md"), x.name),
            )

            # Read markdown_http_base from raw conf.py values, so it works
            # even when sphinx_markdown_builder is not listed in extensions
            # (it is only loaded in the markdown subprocess build).
            http_base = (
                self.app.config._raw_config.get("markdown_http_base")
                or getattr(self.app.config, "markdown_http_base", "")
            ).rstrip("/")

            for md_file in sorted_files:
                # Extract title from the markdown file
                title = self.extract_title_from_markdown(md_file)

                # Create the URL based either on
                # - the relative path from output directory, or
                # - markdown_http_base + the relative path
                rel_path = md_file.relative_to(self.outdir)
                if http_base:
                    url = f"{http_base}/{rel_path}"
                else:
                    url = str(rel_path)

                # Write the link
                sitemap.write(
                    f"- [{title}]({url}): {self.get_page_description(md_file)}\n"
                )

            # Link to llms-full.txt when it was also generated
            if getattr(self.app.config, "llms_txt_full_build", True):
                if http_base:
                    full_url = f"{http_base}/llms-full.txt"
                else:
                    full_url = "llms-full.txt"
                sitemap.write(
                    f"\n---\n\nFor more comprehensive documentation, see [llms-full.txt]({full_url})\n"
                )

            logger.info(f"Created llms.txt sitemap: {llms_txt_path}")

    def extract_title_from_markdown(self, md_file: Path) -> str:
        """Extract the title from a markdown file."""
        try:
            with open(md_file, encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

                # Look for the first heading (starts with #)
                for line in lines:
                    line = line.strip()
                    if line.startswith("#"):
                        title = line.lstrip("#").strip()
                        return title

                # If no heading found, try to get title from filename
                base_name = md_file.stem.replace(".html", "")
                if base_name == "index":
                    return "Home"
                return base_name.replace("_", " ").title()
        except Exception:
            # Fallback to filename without extension
            base_name = md_file.stem.replace(".html", "")
            if base_name == "index":
                return "Home"
            return base_name.replace("_", " ").title()

    def _get_docname_from_md_file(self, md_file: Path) -> str:
        """Return the Sphinx docname for a markdown build output file."""
        rel_path = md_file.relative_to(self.md_build_dir)
        return rel_path.with_suffix("").as_posix()

    def get_page_description(self, md_file: Path) -> str:
        """Get a brief description of the page content.

        If the source page defines an ``html_meta`` description (via
        ``.. meta:: :description:`` in rST or ``html_meta:`` frontmatter in
        MyST), that value is used.  Otherwise the first 100 characters of the
        first meaningful paragraph in the generated markdown are returned.
        """
        docname = self._docname_by_output_file.get(md_file, "")
        if docname:
            try:
                doctree = self.app.env.get_doctree(docname)
                for node in doctree.traverse(docutils.nodes.meta):
                    if node.get("name") == "description" and node.get("content"):
                        return node["content"]
            except Exception:
                logger.exception(
                    "Failed to read html_meta description from doctree for '%s'; "
                    "falling back to content-based description",
                    docname,
                )

        return self.extract_description_from_markdown(md_file)

    @staticmethod
    def extract_description_from_markdown(md_file: Path) -> str:
        """Extract a content-based description from a markdown file.

        Returns the first 100 characters of the first meaningful paragraph,
        or a filename-based fallback if no suitable paragraph is found.
        """
        try:
            with open(md_file, encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
                anchor = re.compile(r"^<a\b[^>]*>\s*</a>$", re.IGNORECASE)

                # Skip HTML comments and look for the first meaningful paragraph
                for line in lines:
                    line = line.strip()
                    # Skip empty lines, headings, anchors, and HTML comments
                    if (
                        line
                        and not line.startswith("#")
                        and not line.startswith("<!--")
                        and not line.startswith("-->")
                        and not line.startswith("..")
                        and not anchor.match(line)
                        and len(line) > 10
                    ):  # Ensure it's substantial content
                        return line[:100] + "..." if len(line) > 100 else line

                # Fallback descriptions based on filename
                base_name = md_file.stem.replace(".html", "")
                if base_name == "index":
                    return "Main documentation page"
                elif base_name == "test":
                    return "Testing and example page"
                else:
                    return "Page content"
        except Exception:
            # Fallback descriptions based on filename
            base_name = md_file.stem.replace(".html", "")
            if base_name == "index":
                return "Main documentation page"
            elif base_name == "test":
                return "Testing and example page"
            else:
                return "Page content"


def setup(app: Sphinx) -> dict[str, Any]:
    """Set up the Sphinx extension."""
    if app.tags.has("sphinx_llm_markdown"):
        app.setup_extension("sphinx_markdown_builder")
    app.add_builder(SphinxLlmMarkdownBuilder)
    app.add_config_value("llms_txt_enabled", True, "")
    app.add_config_value("llms_txt_description", "", "env")
    app.add_config_value("llms_txt_build_parallel", True, "env")
    app.add_config_value("llms_txt_suffix_mode", "auto", "env")
    app.add_config_value("llms_txt_full_build", True, "env")
    app.add_config_value("llms_txt_markdown_layout", "standard", "env")
    app.add_config_value("llms_txt_markdown_root_links", False, "env")
    generator = MarkdownGenerator(app)
    generator.setup()

    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
