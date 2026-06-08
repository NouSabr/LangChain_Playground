import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

def get_summary_prompt_template() -> str:
    template = """ 
        Given the information {information} about a movie, I want you to create:
        1. A short summary
        2. Two interesting facts about it.

        """
    return template

PROMPT_TEMPLATES = {
    "summary": get_summary_prompt_template()
}

llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT,
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,  # type: ignore
)

# embedding_llm = AzureOpenAIEmbeddings(
#     azure_deployment=EMBEDDING_DEPLOYMENT,
#     api_version=EMBEDDING_API_VERSION,
#     azure_endpoint=EMBEDDING_ENDPOINT,
#     api_key=EMBEDDING_API_KEY,  # type: ignore
#     dimensions=1024,
# )


def create_template_prompt(prompt_template:str) -> PromptTemplate:
    template = PROMPT_TEMPLATES.get(prompt_template)
    return PromptTemplate(input_variables=["information"], template = template) # type:ignore

def main():
    information = """
        The Brothers Bloom, orphaned at a young age, grow up in a series of foster homes. Thirteen-year-old Stephen dreams up an elaborate scenario to encourage his younger brother, Bloom, to talk to a girl, and the plan becomes their first confidence trick.

        Twenty-five years later, the brothers are successful con men, and celebrate the end of a job in Berlin with their accomplice and explosives expert, Bang Bang. Bloom longs for an "unwritten life" beyond Stephen's schemes, and the brothers go their separate ways. Three months later, Stephen finds Bloom in Montenegro and convinces him to execute one final con: their target is Penelope Stamp, a wealthy heiress living alone in her New Jersey mansion.

        Bloom inserts himself into Penelope's isolated life by running into her sports car on a bicycle, and they bond over her eccentric array of hobbies, including various musical instruments, chainsaw juggling, and pinhole photography. Exploiting Penelope's loneliness and craving for adventure, Bloom masquerades as an antiques dealer leaving for Europe, and Penelope arrives at the harbor to sail with the brothers and Bang Bang to Greece.

        As part of the con, Melville, a Belgian black marketeer hired by Stephen, tells Penelope that the brothers are smugglers and offers them an illicit job: in exchange for $1 million, he will procure a rare book in Prague for them to sell for $2.5 million. Penelope is thrilled and "convinces" the brothers to accept, and Bloom and Penelope struggle with their mutual attraction, but Stephen warns that the con will fail if Bloom actually falls for Penelope.

        In Prague, Bloom is approached by the brothers' mentor-turned-enemy, Diamond Dog, but Stephen attacks Diamond Dog with a broken bottle, warning him to stay away. Melville disappears with Penelope's $1 million, according to plan, but she is determined to complete the job. The brothers arrange for her to steal the fake book from Prague Castle, where a mixup with Bang Bang's explosives leads to Penelope being caught, but she talks her way out of police custody.

        They go to Mexico to complete the "sale", but Bloom has fallen in love with Penelope and reveals her adventure has been a sham. Preparing to flee together, they are confronted by Stephen; the brothers fight and a gun accidentally discharges, mortally wounding Stephen. Realizing the blood is fake and this was yet another ruse, Penelope leaves broken-hearted. Bloom punches Stephen and leaves for Montenegro once again.

        Three months later, Penelope reunites with Bloom, who is unable to deny his love for her but unwilling to let her become a con artist herself. He meets with Stephen and Bang Bang, preparing to finish the con and fake their own deaths. They go to St. Petersburg, where Stephen has arranged to "sell" the book to Diamond Dog, who will pose as a Russian mobster, but they are ambushed by Diamond Dog's men. Stephen is kidnapped and Bang Bang's car explodes, leaving Penelope and Bloom uncertain whether she has faked her death.

        A $1.75 million ransom demand leads Bloom to suspect this is another of Stephen's tricks, but Penelope wires the money. Arriving at an abandoned theater, Bloom finds Stephen badly beaten and held at gunpoint, and a phone call from Diamond Dog confirms that he has double-crossed the brothers. Bloom shoots first, forcing the gunman to flee, but Stephen takes a bullet for Bloom and collapses. Bloom asks whether this was real, and Stephen leaps to his feet, assuring his brother that he is fine, and tells him to go on the run with Penelope and that they will meet again.

        Driving away with Penelope, Bloom discovers that the bloodstains on his shirt have oxidized, revealing that his brother's blood was real. Stephen dies peacefully, having pulled off "the perfect con", as Bloom and Penelope embark on a new life together.
        """

    summary_prompt_template = create_template_prompt("summary")

    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print (response.content)
    # print("Hello from langchain-playground!")


if __name__ == "__main__":
    main()
