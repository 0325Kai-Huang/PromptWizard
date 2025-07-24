from typing import Dict
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import LLM
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random
from ..base_classes import LLMConfig
from ..constants.str_literals import InstallLibs, OAILiterals, \
    OAILiterals, LLMLiterals, LLMOutputTypes
from .llm_helper import get_token_counter
from ..exceptions import GlueLLMException
from ..utils.runtime_tasks import install_lib_if_missing
from ..utils.logging import get_glue_logger
from ..utils.runtime_tasks import str_to_class
import os
logger = get_glue_logger(__name__)


global_session: Optional[aiohttp.ClientSession] = None
global_connector: Optional[aiohttp.TCPConnector] = None

async def init_global_session():
    """异步初始化全局 session"""
    global global_session, global_connector
    global_connector = aiohttp.TCPConnector(
        limit=settings.CONNECTOR_LIMIT,
        limit_per_host=settings.CONNECTOR_LIMIT_PER_HOST,
        force_close=settings.CONNECTOR_FORCE_CLOSE,
        keepalive_timeout=settings.CONNECTOR_KEEPALIVE_TIMEOUT,
    ) # 此配置没有复现error,用下面的配置项复现
    # global_connector = aiohttp.TCPConnector(
    #     limit=500,
    #     limit_per_host=100,
    #     force_close=False,
    #     keepalive_timeout=3,
    #     enable_cleanup_closed=True,
    # )

    global_session = aiohttp.ClientSession(connector=global_connector)
# 自定义异步模型调用，用于替换openai的调用
async def async_request_post(trace_id, url, request_data, timeout, module, headers=None):
    """
    异步请求抽取公共逻辑
    """
    # ASYNC_RETRIES为重试次数
    for attempt in range(settings.ASYNC_RETRIES):
        try:
            if not global_session:
                await init_global_session()
            # 使用aiohttp发送异步请求
            async with global_session.post(url=url, json=request_data, timeout=timeout, headers=headers) as response:
                response.raise_for_status()
                # 根据响应类型处理数据
                content_type = response.headers.get('Content-Type', '')
                if 'json' in content_type:
                    data = await response.json()
                else:
                    data = await response.text()
                # breakpoint()
                return response.status, data
        except Exception as e:
            if isinstance(e, asyncio.TimeoutError):
                error(f"trace_id: {trace_id}, {module} request timeout")
                raise
            elif isinstance(e, (aiohttp.ClientOSError, ConnectionResetError, aiohttp.ServerDisconnectedError)):
                if attempt == (settings.ASYNC_RETRIES-1):
                    error(f"trace_id: {trace_id}, {module} Connection error: {e}, retry failed")
                    raise
                error(f"trace_id: {trace_id}, {module} Connection error: {e}, attempt: {attempt}, ASYNC_RETRIES: {settings.ASYNC_RETRIES}")
            else:
                error_info = traceback.format_exc()
                error(f"trace_id: {trace_id}, {module} request failed: error message:{error_info}")
                raise


def call_api(messages):

    from openai import OpenAI
    from azure.identity import get_bearer_token_provider, AzureCliCredential
    from openai import AzureOpenAI

    if os.environ['USE_OPENAI_API_KEY'] == "True":
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        response = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL_NAME"],
        messages=messages,
        temperature=0.0,
        )
    else:
        token_provider = get_bearer_token_provider(
                AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
            )
        client = AzureOpenAI(
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_ad_token_provider=token_provider
            )
        response = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            messages=messages,
            temperature=0.0,
        )
        ## 替换自己训练的模型去调用，使得prompt更适配自己的模型
        response = async_request_post(trace_id, url, request_data, timeout, module, headers)

    prediction = response.choices[0].message.content
    return prediction


class LLMMgr:
    @staticmethod
    def chat_completion(messages: Dict):
        llm_handle = os.environ.get("MODEL_TYPE", "AzureOpenAI")
        try:
            if(llm_handle == "AzureOpenAI"): 
                # Code to for calling LLMs
                return call_api(messages)
            elif(llm_handle == "LLamaAML"):
                # Code to for calling SLMs
                return 0
        except Exception as e:
            print(e)
            return "Sorry, I am not able to understand your query. Please try again."
            # raise GlueLLMException(f"Exception when calling {llm_handle.__class__.__name__} "
            #                        f"LLM in chat mode, with message {messages} ", e)
        

    @staticmethod
    def get_all_model_ids_of_type(llm_config: LLMConfig, llm_output_type: str):
        res = []
        if llm_config.azure_open_ai:
            for azure_model in llm_config.azure_open_ai.azure_oai_models:
                if azure_model.model_type == llm_output_type:
                    res.append(azure_model.unique_model_id)
        if llm_config.custom_models:
            if llm_config.custom_models.model_type == llm_output_type:
                res.append(llm_config.custom_models.unique_model_id)
        return res

    @staticmethod
    def get_llm_pool(llm_config: LLMConfig) -> Dict[str, LLM]:
        """
        Create a dictionary of LLMs. key would be unique id of LLM, value is object using which
        methods associated with that LLM service can be called.

        :param llm_config: Object having all settings & preferences for all LLMs to be used in out system
        :return: Dict key=unique_model_id of LLM, value=Object of class llama_index.core.llms.LLM
        which can be used as handle to that LLM
        """
        llm_pool = {}
        az_llm_config = llm_config.azure_open_ai

        if az_llm_config:
            install_lib_if_missing(InstallLibs.LLAMA_LLM_AZ_OAI)
            install_lib_if_missing(InstallLibs.LLAMA_EMB_AZ_OAI)
            install_lib_if_missing(InstallLibs.LLAMA_MM_LLM_AZ_OAI)
            install_lib_if_missing(InstallLibs.TIKTOKEN)

            import tiktoken
            # from llama_index.llms.azure_openai import AzureOpenAI
            from openai import AzureOpenAI
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
            from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal

            az_token_provider = None
            # if az_llm_config.use_azure_ad:
            from azure.identity import get_bearer_token_provider, AzureCliCredential
            az_token_provider = get_bearer_token_provider(AzureCliCredential(),
                                                        "https://cognitiveservices.azure.com/.default")

            for azure_oai_model in az_llm_config.azure_oai_models:
                callback_mgr = None
                if azure_oai_model.track_tokens:
                    
                    # If we need to count number of tokens used in LLM calls
                    token_counter = TokenCountingHandler(
                        tokenizer=tiktoken.encoding_for_model(azure_oai_model.model_name_in_azure).encode
                        )
                    callback_mgr = CallbackManager([token_counter])
                    token_counter.reset_counts()
                    # ()

                if azure_oai_model.model_type in [LLMOutputTypes.CHAT, LLMOutputTypes.COMPLETION]:
                    # ()
                    llm_pool[azure_oai_model.unique_model_id] = \
                        AzureOpenAI(
                            # use_azure_ad=az_llm_config.use_azure_ad,
                                    azure_ad_token_provider=az_token_provider,
                                    # model=azure_oai_model.model_name_in_azure,
                                    # deployment_name=azure_oai_model.deployment_name_in_azure,
                                    api_key=az_llm_config.api_key,
                                    azure_endpoint=az_llm_config.azure_endpoint,
                                    api_version=az_llm_config.api_version,
                                    # callback_manager=callback_mgr
                                    )
                    # ()
                elif azure_oai_model.model_type == LLMOutputTypes.EMBEDDINGS:
                    llm_pool[azure_oai_model.unique_model_id] =\
                        AzureOpenAIEmbedding(use_azure_ad=az_llm_config.use_azure_ad,
                                             azure_ad_token_provider=az_token_provider,
                                             model=azure_oai_model.model_name_in_azure,
                                             deployment_name=azure_oai_model.deployment_name_in_azure,
                                             api_key=az_llm_config.api_key,
                                             azure_endpoint=az_llm_config.azure_endpoint,
                                             api_version=az_llm_config.api_version,
                                             callback_manager=callback_mgr
                                             )
                elif azure_oai_model.model_type == LLMOutputTypes.MULTI_MODAL:

                    llm_pool[azure_oai_model.unique_model_id] = \
                        AzureOpenAIMultiModal(use_azure_ad=az_llm_config.use_azure_ad,
                                              azure_ad_token_provider=az_token_provider,
                                              model=azure_oai_model.model_name_in_azure,
                                              deployment_name=azure_oai_model.deployment_name_in_azure,
                                              api_key=az_llm_config.api_key,
                                              azure_endpoint=az_llm_config.azure_endpoint,
                                              api_version=az_llm_config.api_version,
                                              max_new_tokens=4096
                                              )

        if llm_config.custom_models:
            for custom_model in llm_config.custom_models:
                # try:
                custom_llm_class = str_to_class(custom_model.class_name, None, custom_model.path_to_py_file)

                callback_mgr = None
                if custom_model.track_tokens:
                    # If we need to count number of tokens used in LLM calls
                    token_counter = TokenCountingHandler(
                        tokenizer=custom_llm_class.get_tokenizer()
                        )
                    callback_mgr = CallbackManager([token_counter])
                    token_counter.reset_counts()
                llm_pool[custom_model.unique_model_id] = custom_llm_class(callback_manager=callback_mgr)
                # except Exception as e:
                    # raise GlueLLMException(f"Custom model {custom_model.unique_model_id} not loaded.", e)
        return llm_pool

    @staticmethod
    def get_tokens_used(llm_handle: LLM) -> Dict[str, int]:
        """
        For a given LLM, output the number of tokens used.

        :param llm_handle: Handle to a single LLM
        :return: Dict of token-type and count of tokens used
        """
        token_counter = get_token_counter(llm_handle)
        if token_counter:
            return {
                LLMLiterals.EMBEDDING_TOKEN_COUNT: token_counter.total_embedding_token_count,
                LLMLiterals.PROMPT_LLM_TOKEN_COUNT: token_counter.prompt_llm_token_count,
                LLMLiterals.COMPLETION_LLM_TOKEN_COUNT: token_counter.completion_llm_token_count,
                LLMLiterals.TOTAL_LLM_TOKEN_COUNT: token_counter.total_llm_token_count
                }
        return None
