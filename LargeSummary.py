from abc import ABC, abstractmethod
import json

class LargeSummary(ABC):
    @abstractmethod
    def getSummary(self, path: str, llm: object) -> dict:
        pass

class GenerateReport(LargeSummary):
    def getSummary(self, path, llm):
        from langchain.chains.summarize import load_summarize_chain
        from langchain.prompts import PromptTemplate
        from langchain.output_parsers import StrOutputParser

        docs = []
        with open(path, "r") as file:
            chunk = ""
            for line in file:
                if '===' in line[:3]:
                    docs.append(chunk)
                    chunk = ""
                elif line.strip() == '':
                    continue
                else:
                    chunk += line + '\n'
            docs.append(chunk)

        str_parser = StrOutputParser()
        prompt = PromptTemplate(
            template="""
            Human:
            {instructions} : \"{document}\"
            Assistant:""",
            input_variables=["instructions", "document"]
        )

        summary_chain = prompt | llm | str_parser
        instruction_prompt = """
        You'll be provided multiple sets of log analysis insights...
        Example Output Format: {'Cause': <Cause of Incident>, 'Solution': <Possible solutions to fix the incident>}
        """
        report = summary_chain.invoke({
            "instructions": instruction_prompt,
            "document": '\n'.join(docs)
        })

        filter_report = ""
        flag = False
        for x in report:
            if x == '{':
                flag = True
            if flag:
                filter_report += x
            if x == '}':
                flag = False
                break

        filter_report = json.loads(filter_report)
        return filter_report
