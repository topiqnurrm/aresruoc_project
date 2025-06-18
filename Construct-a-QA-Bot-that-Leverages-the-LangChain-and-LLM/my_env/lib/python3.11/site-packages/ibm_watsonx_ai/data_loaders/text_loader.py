#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

__all__ = ["TextLoader"]

import io
import logging

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING
from queue import Empty

from ibm_watsonx_ai.wml_client_error import MissingExtension
from ibm_watsonx_ai.helpers.remote_document import RemoteDocument

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _sequential_download(q, docs):
    """Helper function for parallel downloading documents in sequence."""
    for doc in docs:
        doc.download()
        q.put(TextLoader(doc).load()[0])


def _asynch_download(args):
    """Helper function for parallel downloading documents (full asynchronous version)."""
    (q_input, qs_output) = args

    while True:
        try:
            i, doc = q_input.get(block=False)
            doc.download()
            qs_output[i].put(TextLoader(doc).load()[0])
        except Empty:
            return


class TextLoader:
    """
    TextLoader class for extraction txt, pdf, html and docx file from bytearray format.

    :param documents: Documents to extraction from bytearray format
    :type documents: RemoteDocument, list[RemoteDocument]

    """

    def __init__(self, documents: list[RemoteDocument] | RemoteDocument) -> None:
        self.files = documents

    def load(self):
        """
        Load text from bytearray data.
        """
        documents = []

        if isinstance(self.files, list):
            with ThreadPoolExecutor(min(5, len(self.files))) as executor:
                future_to_file = {
                    executor.submit(self.process_file, doc): doc for doc in self.files
                }
                for future in as_completed(future_to_file):
                    try:
                        document = future.result()
                        if document:
                            documents.append(document)
                    except Exception as e:
                        logger.error(f"Error processing file: {e}")
        else:
            documents = [self.process_file(self.files)]

        return documents

    def process_file(self, file: RemoteDocument) -> Document | None:
        """
        Process RemoteDocument file to get LangChain's Document object with page_content and metadata attribute
        """
        try:
            from langchain_core.documents import Document as LCDocument
        except ImportError:
            raise MissingExtension("langchain-core")

        file_content = getattr(file, "content", None)
        document_id = getattr(file, "document_id", None)
        file_type = self.identify_file_type(file_content)

        file_type_handlers = {
            "text/plain": self._txt_to_string,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": self._docs_to_string,
            "application/pdf": self._pdf_to_string,
            "text/html": self._html_to_string,
        }

        handler = file_type_handlers.get(file_type, None)

        if handler:
            try:
                text = handler(file_content)
            except Exception as e:
                logger.error(f"Error reading document {document_id}: {e}")
                return None
        else:
            logger.error(
                f"Unsupported file type. Supported file types: {list(file_type_handlers)}"
            )
            return None

        metadata = {
            "document_id": document_id,
        }

        return LCDocument(page_content=text, metadata=metadata)

    def identify_file_type(self, data: bytes | None) -> str | None:
        """
        Identifying file type by bytearray input data
        """
        if data:
            if self._is_pdf(data):
                return "application/pdf"
            elif self._is_html(data):
                return "text/html"
            elif self._is_docx(data):
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif self._is_plain_text(data):
                return "text/plain"
        return None

    @staticmethod
    def _is_pdf(data: bytes) -> bool:
        return data[:4] == b"%PDF"

    @staticmethod
    def _is_html(data: bytes) -> bool:
        try:
            text = data.decode("utf-8", errors="ignore").lower()
            return "<html" in text or "<!doctype html" in text
        except UnicodeDecodeError:
            return False

    @staticmethod
    def _is_docx(data: bytes) -> bool:
        try:
            try:
                from docx import Document as DocxDocument
            except ImportError:
                raise MissingExtension("python-docx")
            from io import BytesIO

            DocxDocument(BytesIO(data))
            return True
        except Exception:
            return False

    @staticmethod
    def _is_plain_text(data: bytes) -> bool:
        try:
            data.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    @staticmethod
    def _txt_to_string(binary_data: bytes) -> str:
        return binary_data.decode("utf-8", errors="ignore")

    @staticmethod
    def _docs_to_string(binary_data: bytes) -> str:
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise MissingExtension("python-docx")

        with io.BytesIO(binary_data) as open_docx_file:
            doc = DocxDocument(open_docx_file)
            full_text = [para.text for para in doc.paragraphs]
            return "\n".join(full_text)

    @staticmethod
    def _pdf_to_string(binary_data: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise MissingExtension("pypdf")

        with io.BytesIO(binary_data) as open_pdf_file:
            reader = PdfReader(open_pdf_file)
            full_text = [page.extract_text() for page in reader.pages]
            return "\n".join(full_text)

    @staticmethod
    def _html_to_string(binary_data: bytes) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise MissingExtension("beautifulsoup4")

        soup = BeautifulSoup(binary_data, "html.parser")
        return soup.get_text()
