
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-small",
    task="text2text-generation",
    model_kwargs={"temperature": 0, "max_length": 256},
)
chat_model = ChatHuggingFace(llm=llm)
print("Model initialized successfully.")



class Agent:
    """Base class for all medical agents."""
    def __init__(self, model, role=None, medical_report=None, extra_info=None):
        self.model = model
        self.role = role
        self.medical_report = medical_report
        self.extra_info = extra_info
        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self):
        """Creates a prompt template based on the agent's role."""
        if self.role == "MultidisciplinaryTeam":
            template_str = f"""
                Act as a multidisciplinary medical board.
                Task: Review all specialist reports, identify the 3 most likely health issues, and explain your reasoning in 1-2 sentences for each.
                Format: Use bullet points: "Health Issue – Reason".

                Cardiologist Report: {self.extra_info.get('Cardiologist', '')}
                Psychologist Report: {self.extra_info.get('Psychologist', '')}
                Pulmonologist Report: {self.extra_info.get('Pulmonologist', '')}
                Neurologist Report: {self.extra_info.get('Neurologist', '')}
                Gastroenterologist Report: {self.extra_info.get('Gastroenterologist', '')}
                Oncologist Report: {self.extra_info.get('Oncologist', '')}
            """
            return PromptTemplate.from_template(template_str)

        templates = {
            "Cardiologist": "You are a Cardiologist. Review the report for heart issues (arrhythmias, ischemia) and suggest next steps. Format: 'Possible Issues: ...' and 'Next Steps: ...'. Patient Report: {medical_report}",
            "Psychologist": "You are a Psychologist. Review the report for signs of anxiety, depression, or stress disorders and suggest interventions. Format: 'Possible Issues: ...' and 'Next Steps: ...'. Patient Report: {medical_report}",
            "Pulmonologist": "You are a Pulmonologist. Review the report for breathing disorders (asthma, COPD) and suggest tests. Format: 'Possible Issues: ...' and 'Next Steps: ...'. Patient Report: {medical_report}",
            "Neurologist": "You are a Neurologist. Review the report for neurological signs (headaches, dizziness) and suggest imaging. Format: 'Possible Issues: ...' and 'Next Steps: ...'. Patient Report: {medical_report}",
            "Gastroenterologist": "You are a Gastroenterologist. Review the report for digestive issues (GERD, IBS) and suggest tests. Format: 'Possible Issues: ...' and 'Next Steps: ...'. Patient Report: {medical_report}",
            "Oncologist": "You are an Oncologist. Review the report for any signs of cancer or tumors that need investigation. Format: 'Possible Issues: ...' and 'Next Steps: ...'. Patient Report: {medical_report}"
        }

        if self.role in templates:
            return PromptTemplate.from_template(templates[self.role])
        else:
            raise ValueError(f"Invalid role specified: {self.role}")

    def run(self):
        """Formats the prompt and invokes the language model."""
        print(f"Running agent: {self.role}...")
        prompt_value = self.prompt_template.format(medical_report=self.medical_report) if self.role != "MultidisciplinaryTeam" else self.prompt_template.format()

        try:
            response = self.model.invoke(prompt_value)
            return response.content
        except Exception as e:
            print(f"Error invoking model for {self.role}: {e}")
            return f"Error generating response for {self.role}."

class Cardiologist(Agent):
    def __init__(self, model, medical_report):
        super().__init__(model, "Cardiologist", medical_report=medical_report)

class Psychologist(Agent):
    def __init__(self, model, medical_report):
        super().__init__(model, "Psychologist", medical_report=medical_report)

class Pulmonologist(Agent):
    def __init__(self, model, medical_report):
        super().__init__(model, "Pulmonologist", medical_report=medical_report)

class Neurologist(Agent):
    def __init__(self, model, medical_report):
        super().__init__(model, "Neurologist", medical_report=medical_report)

class Gastroenterologist(Agent):
    def __init__(self, model, medical_report):
        super().__init__(model, "Gastroenterologist", medical_report=medical_report)

class Oncologist(Agent):
    def __init__(self, model, medical_report):
        super().__init__(model, "Oncologist", medical_report=medical_report)

class MultidisciplinaryTeam(Agent):
    def __init__(self, model, specialist_reports):
        super().__init__(model, "MultidisciplinaryTeam", extra_info=specialist_reports)




report_path = ""

try:
    with open(report_path, "r") as file:
        medical_report_content = file.read()

    # Create instances of all specialized agents
    specialists = {
        "Cardiologist": Cardiologist(chat_model, medical_report_content),
        "Psychologist": Psychologist(chat_model, medical_report_content),
        "Pulmonologist": Pulmonologist(chat_model, medical_report_content),
        "Neurologist": Neurologist(chat_model, medical_report_content),
        "Gastroenterologist": Gastroenterologist(chat_model, medical_report_content),
        "Oncologist": Oncologist(chat_model, medical_report_content)
    }

    # Concurrently run all specialist agents
    specialist_responses = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_agent = {executor.submit(agent.run): name for name, agent in specialists.items()}
        for i, future in enumerate(as_completed(future_to_agent)):
            name = future_to_agent[future]
            response_content = future.result()
            specialist_responses[name] = response_content
            print(f"--- Report from {name} ---\n{response_content}\n")

    # Create and run the MultidisciplinaryTeam agent
    team_agent = MultidisciplinaryTeam(chat_model, specialist_responses)
    final_diagnosis = team_agent.run()

    # Prepare and save the final output
    final_diagnosis_text = "### Final Diagnosis by Multidisciplinary Team:\n\n" + (final_diagnosis or "No diagnosis generated.")
    output_path = "results/final_diagnosis.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as txt_file:
        txt_file.write(final_diagnosis_text)

    print("="*50)
    print(final_diagnosis_text)
    print("="*50)
    print(f"Final diagnosis saved to {output_path}")

except FileNotFoundError:
    print(f"Error: Make sure you have uploaded '{report_path}' to your Colab session!")
    exit()