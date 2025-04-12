import asyncio
import io
import logging
import re
import sys
import threading
import queue
import markdown
import streamlit as st
import html
import datetime

from fpdf import FPDF, HTMLMixin
from fpdf.html import HTML2FPDF

from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete

# note this was loaded into stage
from agent_gateway import Agent
from agent_gateway.tools import PythonTool
from agent_gateway.gateway.constants import END_OF_PLAN, FUSION_FINISH, FUSION_REPLAN



# Get the current Snowflake session
session = get_active_session()

# App title and introduction
st.title("🚗 Auto Manufacturing Agent 🏭  ")
st.markdown("""
### Computer Vision with LandingAI + Graph Reasoning with RelationalAI

**App Usage Notes:**
- This will take 2-3 of minutes when running a question with the RelationalAI dependency for the first time during a session to start up the engine. After this initial question additional questions will answer much faster.")
- Conversation history is not currently implemented in this simple version.
""")


# set up app tabs
tab1, tab2 = st.tabs(["🗒 Instructions", "🤖 Chat with Agent"])


# -------------------------------
# changing planning prompts
# -------------------------------
# see https://github.com/Snowflake-Labs/orchestration-framework/blob/main/agent_gateway/tools/snowflake_prompts.py
#
PLANNING_PROMPT = (
    "Question: Is this part defective?\n"
    "Thought: I need to need to identify if this part is defective.\n"
    '1. ll_inference_tool("")\n'
    "Thought: I can answer the question now by summarizing the output.\n"
    f"3. fuse(){END_OF_PLAN}\n"
    "###\n"
    "Question: What is the downstream impact of this defect?\n"
    "Thought: I need to understand if this part is defective first\n"
    '1. ll_inference_tool("is this part defective")\n'
    "Thought: I do not need to use anymore tolls if the part is not defective. If it is I need to move on.\n"
    "Thought: I need to find which part this associates with.\n"
    '2. get_sku_site_tool()\n'
    "Thought: I need to extract out the $site_name from get_sku_site_tool() results. and only the site_name\n"
    '3. key_value_extract_tool()\n'
    "Thought: I need to run the impact analysis with the $site_name \n"
    '4. get_impact_tool($site_name)\n'
    "Thought: I can now answer the impact.I do not need to run any other tools for basic questions\n"
    f"5. fuse(){END_OF_PLAN}\n"
    "###\n"
    "Question: Provide a report of the effect of this defect\n"
    "Thought: I need to understand if this part is defective first\n"
    '1. ll_inference_tool("is this part defective")\n'
    "Thought: I do not need to use anymore tolls if the part is not defective. If it is I need to move on.\n"
    "Thought: I need to find which part this associates with.\n"
    '2. get_sku_site_tool()\n'
    "Thought: I need to extract out the $site_name from get_sku_site_tool() results. and only the site_name\n"
    '3. key_value_extract_tool()\n'
    "Thought: I need to run the impact analysis with the $site_name \n"
    '4. get_impact_tool($site_name)\n'
    "Thought: I can now answer the impact.\n"
    "Thought: I need to provide a comprehensive analysis of the impact summarizing my findings.\n"
    '5. report_synthesis_tool($get_impact_results) \n'
    "Thought: I can now provide a nicely formatted comprehensive assessment.\n"
    f"6. fuse(){END_OF_PLAN}\n"
    "###\n"
    "Question: What is sku number or site that this is from?\n"
    "Thought: I first need to get the sku part assocation. I don't need to run defect analysis/infereincing. I also do not need to run impact analysis if the user is only asking about a current part or site.\n"
    '1. get_sku_site_tool()\n'
    "Thought: I can answer the question now.\n"
    f"2. fuse(){END_OF_PLAN}\n"
    "Question: Plan a vacation?\n"
    "Thought: This question does not have anything to do with auto manufacturing defect analysis. I will let the user know this is not relevant and say nothing more.\n"
    f"1. fuse(){END_OF_PLAN}\n"
)


OUT_PROMPT = (
    "You must solve the Question. You are given Observations and you can use them to solve the Question. "
    "Then you MUST provide a Thought, and then an Action. Do not use any parenthesis.\n"
    "You will be given a question either some passages or numbers, which are observations.\n\n"
    "Thought step can reason about the observations in 1-2 sentences, and Action can be only one type:\n"
    f" (1) {FUSION_FINISH}(answer): returns the answer and finishes the task using information you found from observations."
    f" (2) {FUSION_REPLAN}: returns the original user's question, clarifying questions or comments about why it wasn't answered, and replans in order to get the information needed to answer the question."
    "\n"
    "Follow the guidelines that you will die if you don't follow:\n"
    "  - Answer should be directly answer the question.\n"
    "  - Thought should be 1-2 sentences.\n"
    "  - Action can only be Finish or Replan\n"
    "  - Action should be Finish if you have enough information to answer the question\n"
    "  - Action Should be Replan if you don't have enough information to answer the question\n"
    "  - You must say <END_OF_RESPONSE> at the end of your response.\n"
    "  - If the user's question is too vague or unclear, say why and ask for clarification.\n"
    "  - If the correct tool is used, but the information does not exist, then let the user know.\n"
    "  - The question should be related to your role as a automobile parts manufacturer analyst. If the question is not related to this, kindly decline.\n"
    "  - ONLY RUN THE report_synthesis_tool if the user asks for a report \n"
    "\n"
    "\n"
    "Here are some examples:\n"
    "\n"
    "\n"
    "Question: Is this part defective? \n"
    "landing_lens_inference()\n"
    "Observation: '{backbonepredictions: null,backbonetype: null, latency: {infer_s: 0.2686600685119629,input_conversion_s: 0.0008308887481689453,model_loading_s: 0.0001480579376220703, postprocess_s: 8.082389831542969e-05, preprocess_s: 0.004394054412841797, serialize_s: 0.0001823902130126953}, model_id: a4d905ed-789f-4d09-9a5b-579d6c3aaf46, predictions: {labelIndex: 1, labelName: defects, score: 0.9991890788078308},type: ClassificationPrediction}\n" 
    "\n"
    "Thought: The selected part was found to be defective, with a proability of 99.919% \n"
    f"Action: {FUSION_FINISH}(The selected part was found to be defective, with a proability of 99.919% .)\n"
    "<END_OF_RESPONSE>\n"
    "\n"
    "\n"
    "Question: Plan a trip\n"
    "Thought: I do not need to run any tools and will let the user know this is not relevant% \n"
    f"Action: {FUSION_FINISH}(I am sorry but this question is not revelvant to defect analysis .)\n"
    "<END_OF_RESPONSE>\n"
    "\n"
    "\n"
    "Question: What is the impact of this defect?\n"
    "get_impact($Site)\n"
    "Observation:'{'sku_id': {0: 'SKU201', 1: 'SKU201'}, 'site_id': {0: 'SITE301',1: 'SITE302'},'sku': {0: 'example Car',1: 'example Car'},'site': {0: 'sw RDC',1: 'ne RDC'}}'\n"
    "Thought: I need to provide a thorough analysis of these impacted sites and parts. I'll include a full summary in a well structured and formatted output, including using markdown and bullets under headings \n"
    f"Action: {FUSION_FINISH}(Based on a thorough review of your supply chain, I found the following impacts.)\n"
    "<END_OF_RESPONSE>\n"
    "\n"
    "\n"
    "Question: Prepare a report of your findings\n"
    "report_synthesis_tool(get_impact($Site))\n"
    "Observation:'{'sku_id': {0: 'SKU201', 1: 'SKU201'}, 'site_id': {0: 'SITE301',1: 'SITE302'},'sku': {0: 'example Car',1: 'example Car'},'site': {0: 'sw RDC',1: 'ne RDC'}}'\n"
    "Thought: I need to provide a thorough analysis of these impacted sites and parts. I'll include a full summary in a well structured and formatted output, including using markdown and bullets under headings \n"
    f"Action: {FUSION_FINISH}(Based on a thorough review of your supply chain, I found the following impacts.)\n"
    "<END_OF_RESPONSE>\n"
    "\n"
    "\n"
    "Question: What is the impact of this defect?\n"
    "landing_lens_inference()\n"
    "Observation: '{backbonepredictions: null,backbonetype: null, latency: {infer_s: 0.2686600685119629,input_conversion_s: 0.0008308887481689453,model_loading_s: 0.0001480579376220703, postprocess_s: 8.082389831542969e-05, preprocess_s: 0.004394054412841797, serialize_s: 0.0001823902130126953}, model_id: a4d905ed-789f-4d09-9a5b-579d6c3aaf46, predictions: {labelIndex: 1, labelName: no-defects, score: 0.9991890788078308},type: ClassificationPrediction}\n" 
    "Thought: This part was found to be a no-defect or not defective. Therefore I DO NOT NEED TO RUN ANY MORE DOWNSTREAM ANALYSIS \n"
    f"Action: {FUSION_FINISH}(Since this part was not defective do not run any more tools.)\n"
    "<END_OF_RESPONSE>\n"
    "\n"
    "\n",
)


# -------------------------------
# Defining functions (tools)
# -------------------------------

# this one we are simplifying to have the file match the file selected by user.
# The agent will select to use this tool but ideally it should reason to know what to plug in as param 
def landing_lens_inference(file):
    """Tool to run defect inferencing on a file using Landing Lens"""
    logging.info("Running landing_lens_inference tool")
    sql_cmd = f"""
    SELECT
    LANDINGLENS__VISUAL_AI_PLATFORM_NON_COMMERCIAL_USE.core.run_inference('{file}'
        , '12987d0c-6915-4b79-9e25-cabecd2fe85f') as inference
    """
    inference_out = session.sql(sql_cmd).to_pandas()['INFERENCE'][0]
    return inference_out


# this is just a query to align images to the parts (glue between image selected and parts, since we don't have in the RAI model yet)
# for demo purposes we are just keeping this one static, but this could be more dynamic with some additional tooling
# despite this, as long as the agent knows to use this is all that matters
def get_sku_site():
    """Query defective images to find the site and part"""
    logging.info("Running query_part tool")
    sql_cmd = f"""
    select  sk.name as SKU_NAME, s.name AS SITE_NAME
    from RAI_DEMO.RAI_LAI_MANUFACTURING.IMAGES i
    left join RAI_DEMO.RAI_LAI_MANUFACTURING.SITE s on s.id = i.site
    left join RAI_DEMO.RAI_LAI_MANUFACTURING.SKU sk on sk.id = i.sku
    where i.file_key = '{selected_image}'
    """
    df = session.sql(sql_cmd).to_pandas()
    return df.to_dict()


def key_value_extractor(dict_output):
    """Extract out the value associated with a \key from a dictionary"""
    logging.info("Running key_value_extractor tool")
    return Complete(model, f"#INSTRUCTIONS: You are to extract the site name value from the following. Only include the value. #INPUT: {dict_output}")


def report_synthesis(input):
    """Create a summary including the results="""
    logging.info("Running report_synthesis tool")
    return Complete(model, f"#INSTRUCTIONS: Provide an impact analysis of the impacted sites, locations USING BULLETS. INCLUEDE IN YOUR REPONSE: Based on a comprehensive graph reasoning of your entire car manufacturing supply chain network, we have identified that this defective part will impact car shipment to the following regions:  {input}")


def defect_action(input):
    """take an action based on the downstream impact"""
    logging.info("write defect results")

    session.sql('use schema ')
    # Create a sample DataFrame.
    data = [
        (1, selected_image, input),
    ]
    columns = ["ID", "IMAGE", "DEFECT_NOTES"]
    df = session.create_dataframe(data, schema=columns)
    
    # Write the DataFrame to a Snowflake table in append mode.
    df.write.mode("append").save_as_table("DEFECT_LOG")
    
    return 'results written'


# note that the streamlit is having an issue with the stored proc call (Works fine in notebooks)
# So we created a stored proc that runs the rai model dynamicaly and caches to a table
# then we pull from the table on the fly. This effectively does the same process with more latency to write the results to a table first and then pull
# regardless this effectively demonstrates running the RAI model on the fly and incorporating the results for the agent
@st.cache_data
def get_impact(site_name):
    """
    This query will call the RAI stored proc for a given site name that the agent identifies from the parts.

    Returns a dictionary of the results
    """
    logging.info("getting impact via RAI")
    session.sql(f"CALL RAI_DEMO.RAI_LAI_MANUFACTURING.impact_analysis_cache('Brake Disc', '{site_name}', 'RAI_DEMO.RAI_LAI_MANUFACTURING.GET_IMPACT_RESULT')").collect()
    df = session.sql("select * from RAI_DEMO.RAI_LAI_MANUFACTURING.GET_IMPACT_RESULT").to_pandas()
    return df.to_dict()


# Monkey-patch HTML2FPDF so that its unescape method is available.
HTML2FPDF.unescape = staticmethod(html.unescape)

class MyFPDF(FPDF, HTMLMixin):
    pass

def generate_pdf_report(content, additional_text=None):
    """
    Generates a PDF report from markdown-formatted content.
    Bullet symbols (•) are replaced with a dash (-) for better formatting.
    The report includes a title, subtitle, and current date/time.
    Returns the PDF as a bytes object.
    """
    # Replace bullet symbols with a dash.
    content = content.replace("•", "-")
    if additional_text:
        additional_text = additional_text.replace("•", "-")
    
    # Convert markdown content to HTML using extensions for line breaks.
    html_content = markdown.markdown(content, extensions=['extra', 'nl2br'])
    html_content = html_content.replace("</p>", "</p><br><br>")
    
    if additional_text:
        additional_html = markdown.markdown(additional_text, extensions=['extra', 'nl2br'])
        additional_html = additional_html.replace("</p>", "</p><br><br>")
        html_content += "<br/><br/>" + additional_html

    pdf = MyFPDF()
    pdf.add_page()
    
    # Report header details.
    report_title = "Defect Impact Analysis Report"
    report_subtitle = "Detailed Impact Report Generated with Graph Reasoning"
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add header to the PDF using built-in Helvetica.
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, report_title, ln=True, align='C')
    pdf.set_font("Helvetica", '', 14)
    pdf.cell(0, 10, report_subtitle, ln=True, align='C')
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 10, current_datetime, ln=True, align='C')
    pdf.ln(10)
    
    # Set the main font for the content.
    pdf.set_font("Helvetica", size=12)
    
    # Render the HTML content into the PDF.
    pdf.write_html(html_content)
    
    # Generate the PDF output as bytes.
    pdf_bytes = pdf.output(dest="S").encode("latin1", errors="replace")
    return pdf_bytes


# -------------------------------
# Configure and instantiate tools
# -------------------------------

ll_inference_config = {
    "tool_description": "produces a classification for defects given an image",
    "output_description": "dictionary of defect prediction metadata",
    "python_func": landing_lens_inference,
}


get_sku_site_config = {
    "tool_description": "query to find the sku and site. You will use this to extract the sku name and site name for insights",
    "output_description": "dict of summary data",
    "python_func": get_sku_site,
}

key_value_config = {
    "tool_description": "Extract out the site value results",
    "output_description": "dict of data",
    "python_func": key_value_extractor,
}

get_impact_config = {
    "tool_description": "finds the downstream supply chain impact using a graph analytics.", # Get the sku name first from the key value config",
    "output_description": "dict of data",
    "python_func": get_impact,
}

report_synthesis_config = {
    "tool_description": "prepare a report of your findings", 
    "output_description": "string of a comprehensive analysis findings",
    "python_func": report_synthesis,
}

defect_action_config = {
    "tool_description": "ONLY PERFORM FOR PARTS THAT ARE DEFECTIVE!!! If a downstream impact is identified take action including logging the analysis",
    "output_description": "string of results written",
    "python_func": defect_action,
}

# tool configuration
ll_inference_tool = PythonTool(**ll_inference_config)
get_sku_site_tool = PythonTool(**get_sku_site_config)
key_value_extract_tool = PythonTool(**key_value_config)
get_impact_tool = PythonTool(**get_impact_config)
report_synthesis_tool = PythonTool(**report_synthesis_config)
defect_action_tool = PythonTool(**defect_action_config)


snowflake_tools = [
    ll_inference_tool, 
    get_sku_site_tool, 
    key_value_extract_tool,
    get_impact_tool,
    report_synthesis_tool,
    #defect_action_tool
    
]

# Create the agent
agent = Agent(snowflake_connection=session, tools=snowflake_tools, max_retries=5, planner_example_prompt=PLANNING_PROMPT , fusion_prompt=OUT_PROMPT)

with tab1:
    st.markdown("""

    ### Instructions
    This app demonstrates the art of the possible by leveraging the Agent framework to orchestrate tasks using Snowflake Native Apps as tools.
    The app leverages the [orchestration-framework](https://github.com/Snowflake-Labs/orchestration-framework) developed by SIT.

    This premise is to show how an agent can reason in a manufacturing setting to identify defects on a part and to analyze the impact on the supply chain.

    - 🖼 **LandingAI** is used as a tool for defect classification
    - 🧠 **RelationalAI** is used as a tool for graph analysis using a graph algorithm to find the reach of the defect in the supply chain
    - 🛠️ Additional Python tools are used to enable insights and to aid in agentic reasoning


    #### Example Questions to Ask
    **Basic Questions**
    - What is the sku associated with this part?
    - What site is this from?

    **Landing AI Question**
    - Is this part defective?
    - What is the defect score of this part?

    **Advanced Questions**
    - What is the supply chain impact of this defect?
    
    """
               )

with tab2:
    # -------------------------------
    # User Selections: Model & Image
    # -------------------------------
    model = 'claude-3-5-sonnet'
    
    selected_image = st.selectbox(
        "Select an image",
        [   'defects/cast_def_0_33.jpeg',
            'defects/cast_def_0_40.jpeg',
            'defects/cast_def_0_0.jpeg',
            'defects/cast_def_0_105.jpeg',
            'defects/cast_def_0_107.jpeg',
            'defects/cast_def_0_11.jpeg',
            'defects/cast_def_0_110.jpeg',
            'defects/cast_def_0_118.jpeg',
            'defects/cast_def_0_133.jpeg',
            'defects/cast_def_0_144.jpeg',
            'defects/cast_def_0_148.jpeg',
            'no-defects/cast_ok_0_102.jpeg'
        ]
    )
    
    # Build the file path using the selected image
    file = f'@llens_sample_ds_manufacturing.ball_bearing.dataset/data/{selected_image}'
    
    # Retrieve and display the image with a spinner
    with st.spinner("Loading image..."):
        image_bytes = session.file.get_stream(file, decompress=False).read()
        st.image(image_bytes, caption=selected_image)
    
    # -------------------------------
    # Helper function to extract tool name from log messages
    # -------------------------------
    def extract_tool_name(statement):
        match = re.search(r"Running\s+(.*?)\s+tool", statement)
        if match:
            return match.group(1)
        return "Unknown Tool"
    
    # -------------------------------
    # Setup logging handler to capture tool usage messages
    # -------------------------------
    class QueueLogHandler(logging.Handler):
        def __init__(self, log_queue):
            super().__init__()
            self.log_queue = log_queue
        def emit(self, record):
            msg = self.format(record)
            if "Running" in msg and "tool" in msg:
                tool_name = extract_tool_name(msg)
                self.log_queue.put({"tool": tool_name})
    
    log_queue = queue.Queue()
    queue_handler = QueueLogHandler(log_queue)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    queue_handler.setFormatter(formatter)
    logging.getLogger().addHandler(queue_handler)
    logging.getLogger().setLevel(logging.INFO)
    
    # -------------------------------
    # Conversation Interface with Tool Tracking & Cleaned Output
    # -------------------------------
    if st.button("Start Over"):
        st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]
        st.rerun()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]
    
    st.subheader("Conversation with Agent")
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    prompt = st.chat_input("Type your message")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        msg_queue = queue.Queue()
        # Local list to collect tools used for this specific message.
        message_tools = []
        # A temporary container for live tool updates.
        tool_container = st.empty()
    
        def run_agent_acall():
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(agent.acall(f"{prompt} .Use this file: '{file}'"))
            loop.close()
            sys.stdout = old_stdout
    
            output = new_stdout.getvalue()
            for line in output.splitlines():
                if "Running" in line and "tool" in line:
                    tool_name = extract_tool_name(line)
                    msg_queue.put({"tool": tool_name})
            msg_queue.put({"response": response})
    
        thread = threading.Thread(target=run_agent_acall)
        thread.start()
    
        with st.chat_message("assistant"), st.spinner("Awaiting Response..."):
            final_response = None
            while True:
                try:
                    while not msg_queue.empty():
                        msg = msg_queue.get_nowait()
                        if "tool" in msg:
                            tool_used = msg['tool']
                            message_tools.append(tool_used)
                            tool_container.markdown(f"**Using tool:** {tool_used}")
                        elif "response" in msg:
                            final_response = msg["response"]
                            break
                    while not log_queue.empty():
                        log_msg = log_queue.get_nowait()
                        message_tools.append(log_msg['tool'])
                        tool_container.markdown(f"**Using tool (log):** {log_msg['tool']}")
                    if final_response is not None:
                        break
                except queue.Empty:
                    continue
            thread.join()
            tool_container.empty()
    
            # Final response handling for this message.
            if isinstance(final_response, dict):
                output_text = final_response.get("output", "")
                # Use message_tools (tools used for this message) for the expander.
                st.markdown(f"**Final Output:** {output_text}")
                with st.expander("Tools Used for this Message", expanded=True):
                    if message_tools:
                        for tool in message_tools:
                            st.markdown(f"- **{tool}**")
                    else:
                        st.markdown("No tools were used for this message.")
                st.session_state.messages.append({"role": "assistant", "content": output_text})
    
                # If report_synthesis_tool was used, offer a PDF download.
                if any("report_synthesis" in tool for tool in message_tools):
                    pdf_bytes = generate_pdf_report(output_text)
                    st.download_button(
                        label="Download Impact Report PDF",
                        data=pdf_bytes,
                        file_name="impact_report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})

