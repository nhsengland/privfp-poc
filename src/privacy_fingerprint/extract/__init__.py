from abc import ABC

from privacy_fingerprint.common import Record


class Extractor(ABC):
    def extract_record(self, clinical_text: str) -> Record:
        pass
