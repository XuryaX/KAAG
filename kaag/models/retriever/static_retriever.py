from typing import Dict, Any
from .base import BaseRetriever

class StaticDBRetriever(BaseRetriever):
    def __init__(self):
        self.knowledge_base = {
            "Lightning Network": "A second-layer payment protocol that operates on top of a blockchain-based cryptocurrency.",
            "Blockchain": "A distributed ledger technology that maintains a growing list of records, called blocks, that are securely linked using cryptography.",
            "Fintech": "Financial technology, referring to new tech that seeks to improve and automate the delivery and use of financial services.",
            "Micropayments": "Financial transactions involving very small sums of money.",
            "Scalability": "The capability of a system to handle a growing amount of work, or its potential to be enlarged to accommodate that growth.",
            "Liquidity": "The degree to which an asset or security can be quickly bought or sold in the market without affecting the asset's price.",
            "Implementation": "The process of putting a decision or plan into effect; execution.",
            "API": "Application Programming Interface, a set of functions and procedures allowing the creation of applications that access the features or data of an operating system, application, or other service.",
            "Security": "Measures taken to protect a user or system from theft or damage to hardware, software, or information, as well as from disruption or misdirection of services.",
            "Integration": "The process of bringing together smaller components into a single system that functions as one.",
            "Startup": "A company or project initiated by an entrepreneur to seek, develop, and validate a scalable business model.",
            "CTO": "Chief Technology Officer, an executive-level position focused on scientific and technological issues within an organization.",
            "ROI": "Return on Investment, a performance measure used to evaluate the efficiency of an investment or compare the efficiency of a number of different investments.",
        }

    def retrieve(self, query: str) -> Dict[str, Any]:
        relevant_info = {}
        for key, value in self.knowledge_base.items():
            if query.lower() in key.lower() or query.lower() in value.lower():
                relevant_info[key] = value
        return {"results": relevant_info}