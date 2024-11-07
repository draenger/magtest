import pandas as pd
import json
import os


class BBHFileDataLoader:
    def load_data(self, file_path: str, category: str, data_type: str) -> pd.DataFrame:
        if file_path.endswith(".json"):
            return self._load_json_data(file_path, category, data_type)
        elif file_path.endswith(".txt"):
            return self._load_txt_data(file_path, category, data_type)
        else:
            raise ValueError(f"Unsupported file extension for file: {file_path}")

    def _load_json_data(
        self, file_path: str, category: str, data_type: str
    ) -> pd.DataFrame:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)

            for example in task_data.get("examples", []):
                data.append(
                    {
                        "question": example.get("input", ""),
                        "answer": example.get("target", ""),
                        "explanation": "",
                        "helper_text": "",
                        "category": category,
                        "data_type": data_type,
                    }
                )
        return pd.DataFrame(data)

    def _load_txt_data(
        self, file_path: str, category: str, data_type: str
    ) -> pd.DataFrame:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split on ----- and take second part
        parts = content.split("-----")
        if len(parts) < 2:
            return pd.DataFrame()

        # Split content into helper_text and questions
        content_parts = parts[1].strip().split("\n\n")
        helper_text = content_parts[0].strip()
        question_blocks = content_parts[1:]

        for question_block in question_blocks:
            if "Q:" not in question_block or "A:" not in question_block:
                continue

            # Split into question and answer parts
            qa_parts = question_block.split("A:")
            if len(qa_parts) != 2:
                continue

            # Get question content (remove Q: prefix)
            question = qa_parts[0].strip().split("Q:")[1].strip()

            # Get answer content and extract final answer
            answer_content = qa_parts[1].strip()
            if "So the answer is" in answer_content:
                explanation, answer = answer_content.split("So the answer is")
                explanation = explanation.strip()
                answer = answer.strip()
                # Remove trailing period if exists
                if answer.endswith("."):
                    answer = answer[:-1].strip()
            else:
                explanation = answer_content
                answer = ""

            # print("--------------------------------")
            # print(f"Helper text: \n{helper_text}")
            # print("++++++++++++++++++++++++++++++++")
            # print(f"Question block: \n{question_block}")
            # print("++++++++++++++++++++++++++++++++")
            # print(f"Question: \n{question}")
            # print(f"Explanation: \n{explanation}")
            # print(f"Answer: \n{answer}")
            # print("--------------------------------")
            # print("\n\n")

            if question and answer:
                data.append(
                    {
                        "question": question,
                        "answer": answer,
                        "explanation": explanation.strip(),
                        "helper_text": helper_text,
                        "category": category,
                        "data_type": data_type,
                    }
                )

        return pd.DataFrame(data)
