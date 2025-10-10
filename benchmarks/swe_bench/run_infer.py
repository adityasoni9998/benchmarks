from __future__ import annotations

import json
import os
import toml

from pydantic import SecretStr, BaseModel, Field, SecretStr, ValidationError
from typing import Any

from benchmarks.utils.agent_server import run_remote_evaluation
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.run_evaluation import (
    construct_eval_output_dir,
    create_workspace_for_instance,
    get_instruction,
    make_metadata,
    process_instance_simplified,
)
from benchmarks.utils.runtime import Runtime
from openhands.sdk import LLM, get_logger



class LLMConfig(BaseModel):
    """Configuration for the LLM model.

    Attributes:
        model: The model to use.
        api_key: The API key to use.
        base_url: The base URL for the API. This is necessary for local LLMs.
        api_version: The version of the API.
        aws_access_key_id: The AWS access key ID.
        aws_secret_access_key: The AWS secret access key.
        aws_region_name: The AWS region name.
        num_retries: The number of retries to attempt.
        retry_multiplier: The multiplier for the exponential backoff.
        retry_min_wait: The minimum time to wait between retries, in seconds. This is exponential backoff minimum. For models with very low limits, this can be set to 15-20.
        retry_max_wait: The maximum time to wait between retries, in seconds. This is exponential backoff maximum.
        timeout: The timeout for the API.
        max_message_chars: The approximate max number of characters in the content of an event included in the prompt to the LLM. Larger observations are truncated.
        temperature: The temperature for the API.
        top_p: The top p for the API.
        custom_llm_provider: The custom LLM provider to use. This is undocumented in openhands, and normally not used. It is documented on the litellm side.
        max_input_tokens: The maximum number of input tokens. Note that this is currently unused, and the value at runtime is actually the total tokens in OpenAI (e.g. 128,000 tokens for GPT-4).
        max_output_tokens: The maximum number of output tokens. This is sent to the LLM.
        input_cost_per_token: The cost per input token. This will available in logs for the user to check.
        output_cost_per_token: The cost per output token. This will available in logs for the user to check.
        ollama_base_url: The base URL for the OLLAMA API.
        drop_params: Drop any unmapped (unsupported) params without causing an exception.
        modify_params: Modify params allows litellm to do transformations like adding a default message, when a message is empty.
        disable_vision: If model is vision capable, this option allows to disable image processing (useful for cost reduction).
        caching_prompt: Use the prompt caching feature if provided by the LLM and supported by the provider.
        log_completions: Whether to log LLM completions to the state.
        log_completions_folder: The folder to log LLM completions to. Required if log_completions is True.
        custom_tokenizer: A custom tokenizer to use for token counting.
        native_tool_calling: Whether to use native tool calling if supported by the model. Can be True, False, or not set.
        reasoning_effort: The effort to put into reasoning. This is a string that can be one of 'low', 'medium', 'high', or 'none'. Exclusive for o1 models.
    """

    model: str = Field(default='claude-3-7-sonnet-20250219')
    api_key: SecretStr | None = Field(default=None)
    base_url: str | None = Field(default=None)
    api_version: str | None = Field(default=None)
    aws_access_key_id: SecretStr | None = Field(default=None)
    aws_secret_access_key: SecretStr | None = Field(default=None)
    aws_region_name: str | None = Field(default=None)
    openrouter_site_url: str = Field(default='https://docs.all-hands.dev/')
    openrouter_app_name: str = Field(default='OpenHands')
    # total wait time: 5 + 10 + 20 + 30 = 65 seconds
    num_retries: int = Field(default=4)
    retry_multiplier: float = Field(default=2)
    retry_min_wait: int = Field(default=5)
    retry_max_wait: int = Field(default=30)
    timeout: int | None = Field(default=None)
    max_message_chars: int = Field(
        default=30_000
    )  # maximum number of characters in an observation's content when sent to the llm
    temperature: float = Field(default=0.0)
    top_p: float = Field(default=1.0)
    top_k: int | None = Field(default=None)
    custom_llm_provider: str | None = Field(default=None)
    max_input_tokens: int | None = Field(default=None)
    max_output_tokens: int | None = Field(default=None)
    input_cost_per_token: float | None = Field(default=None)
    output_cost_per_token: float | None = Field(default=None)
    ollama_base_url: str | None = Field(default=None)
    log_completions: bool = Field(default=False)
    # This setting can be sent in each call to litellm
    drop_params: bool = Field(default=True)
    # Note: this setting is actually global, unlike drop_params
    modify_params: bool = Field(default=True)
    disable_vision: bool | None = Field(default=None)
    caching_prompt: bool = Field(default=True)
    custom_tokenizer: str | None = Field(default=None)
    native_tool_calling: bool | None = Field(default=None)
    reasoning_effort: str | None = Field(default='high')

    model_config = {'extra': 'forbid'}

    @classmethod
    def from_toml_section(cls, data: dict) -> dict[str, LLMConfig]:
        """
        Create a mapping of LLMConfig instances from a toml dictionary representing the [llm] section.

        The default configuration is built from all non-dict keys in data.
        Then, each key with a dict value (e.g. [llm.random_name]) is treated as a custom LLM configuration,
        and its values override the default configuration.

        Example:
        Apply generic LLM config with custom LLM overrides, e.g.
            [llm]
            model=...
            num_retries = 5
            [llm.claude]
            model="claude-3-5-sonnet"
        results in num_retries APPLIED to claude-3-5-sonnet.

        Returns:
            dict[str, LLMConfig]: A mapping where the key "llm" corresponds to the default configuration
            and additional keys represent custom configurations.
        """

        # Initialize the result mapping
        llm_mapping: dict[str, LLMConfig] = {}

        # Extract base config data (non-dict values)
        base_data = {}
        custom_sections: dict[str, dict] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                custom_sections[key] = value
            else:
                base_data[key] = value

        # Try to create the base config
        try:
            base_config = cls.model_validate(base_data)
            llm_mapping['llm'] = base_config
        except ValidationError:
            print(
                'Cannot parse [llm] config from toml. Continuing with defaults.'
            )
            # If base config fails, create a default one
            base_config = cls()
            # Still add it to the mapping
            llm_mapping['llm'] = base_config

        # Process each custom section independently
        for name, overrides in custom_sections.items():
            try:
                # Merge base config with overrides
                merged = {**base_config.model_dump(), **overrides}
                custom_config = cls.model_validate(merged)
                llm_mapping[name] = custom_config
            except ValidationError:
                print(
                    f'Cannot parse [{name}] config from toml. This section will be skipped.'
                )
                # Skip this custom section but continue with others
                continue

        return llm_mapping

    def model_post_init(self, __context: Any):
        """Post-initialization hook to assign OpenRouter-related variables to environment variables.

        This ensures that these values are accessible to litellm at runtime.
        """
        super().model_post_init(__context)

        # Assign OpenRouter-specific variables to environment variables
        if self.openrouter_site_url:
            os.environ['OR_SITE_URL'] = self.openrouter_site_url
        if self.openrouter_app_name:
            os.environ['OR_APP_NAME'] = self.openrouter_app_name

        # Assign an API version for Azure models
        # While it doesn't seem required, the format supported by the API without version seems old and will likely break.
        # Azure issue: https://github.com/All-Hands-AI/OpenHands/issues/6777
        if self.model.startswith('azure') and self.api_version is None:
            self.api_version = '2024-08-01-preview'

def get_llm_config_arg(
    llm_config_arg: str, toml_file: str = 'config.toml'
) -> LLMConfig | None:
    """Get a group of llm settings from the config file.

    A group in config.toml can look like this:

    ```
    [llm.gpt-3.5-for-eval]
    model = 'gpt-3.5-turbo'
    api_key = '...'
    temperature = 0.5
    num_retries = 8
    ...
    ```

    The user-defined group name, like "gpt-3.5-for-eval", is the argument to this function. The function will load the LLMConfig object
    with the settings of this group, from the config file, and set it as the LLMConfig object for the app.

    Note that the group must be under "llm" group, or in other words, the group name must start with "llm.".

    Args:
        llm_config_arg: The group of llm settings to get from the config.toml file.
        toml_file: Path to the configuration file to read from. Defaults to 'config.toml'.

    Returns:
        LLMConfig: The LLMConfig object with the settings from the config file.
    """
    # keep only the name, just in case
    llm_config_arg = llm_config_arg.strip('[]')

    # truncate the prefix, just in case
    if llm_config_arg.startswith('llm.'):
        llm_config_arg = llm_config_arg[4:]

    print(f'Loading llm config from {llm_config_arg}')

    # load the toml file
    try:
        with open(toml_file, 'r', encoding='utf-8') as toml_contents:
            toml_config = toml.load(toml_contents)
    except FileNotFoundError as e:
        print(f'Config file not found: {e}')
        return None
    except toml.TomlDecodeError as e:
        print(
            f'Cannot parse llm group from {llm_config_arg}. Exception: {e}'
        )
        return None

    # update the llm config with the specified section
    if 'llm' in toml_config and llm_config_arg in toml_config['llm']:
        return LLMConfig(**toml_config['llm'][llm_config_arg])
    print(f'Loading from toml failed for {llm_config_arg}')
    return None

logger = get_logger(__name__)


def read_completed_instances(output_file: str) -> set:
    """Read completed instance IDs from existing output file."""
    completed = set()
    if not os.path.exists(output_file):
        return completed
        
    try:
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    result = json.loads(line)
                    if 'instance_id' in result:
                        completed.add(result['instance_id'])
    except Exception as e:
        logger.warning(f"Error reading existing output file {output_file}: {e}")
    return completed


def run_local_mode(args, llm, metadata):
    """Run evaluation using local runtime mode (original behavior)."""
    logger.info("Running in LOCAL mode")
    
    # Global variables for runtime methods
    global instances, output_file, results
    instances = None
    output_file = None
    results = []

    def initialize_runtime():
        """Initialize the runtime and retrieve instances to process."""
        global instances, output_file
        output_file = os.path.join(metadata.eval_output_dir or ".", "output.jsonl")

        # Prepare output file directory
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only create directory if dirname is not empty
            os.makedirs(output_dir, exist_ok=True)

        # Read existing completed instances instead of overwriting
        completed_instances = read_completed_instances(output_file)
        if completed_instances:
            logger.info(f"Found {len(completed_instances)} already completed instances")
        else:
            logger.info("No existing results found, starting fresh")
            # Create empty output file only if it doesn't exist
            if not os.path.exists(output_file):
                with open(output_file, "w"):
                    pass

        # Retrieve instances to process, excluding completed ones
        instances = get_dataset(
            metadata.dataset or "",
            metadata.data_split or "",
            output_file,
            metadata.eval_n_limit or 0,
            completed_instances,
        )
        print(f"### OUTPUT FILE: {output_file} ###")
        return instances

    def process_instance(instance):
        """Process a single instance."""
        global results, output_file
        logger.info(f"Processing instance {instance.instance_id}")

        # Create workspace and get actual path
        workspace_path = create_workspace_for_instance(instance, metadata)
        instruction = get_instruction(
            instance, metadata, workspace_path, metadata.prompt_path or ""
        )
        result = process_instance_simplified(instance, instruction, metadata, workspace_path)

        # Save result using the complete format
        result_dict = result.model_dump(mode="json")
        if result.error:
            result_dict["error"] = result.error
        results.append(result_dict)

        logger.info(f"Writing result for {instance.instance_id} to {output_file}")
        logger.info(f"Result dict keys: {list(result_dict.keys())}")
        git_patch_len = len(result_dict.get("test_result", {}).get("git_patch", ""))
        logger.info(f"Git patch length: {git_patch_len}")

        # Write to output file
        assert output_file is not None
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "a") as f:
            json_line = json.dumps(result_dict) + "\n"
            f.write(json_line)
            f.flush()  # Ensure it's written immediately
            logger.info(
                f"Successfully wrote {len(json_line)} characters to output file"
            )

    def complete_runtime():
        """Complete the runtime - any cleanup if needed."""
        logger.info("Runtime completed successfully!")

    # Create and run the Runtime
    runtime = Runtime(
        metadata=metadata,
        initialize_runtime=initialize_runtime,
        process_instance=process_instance,
        complete_runtime=complete_runtime,
        num_workers=args.eval_num_workers,
    )

    runtime.run()


def main():
    default_prompt_path = os.path.join(
        os.path.dirname(__file__), "prompts", "default.j2"
    )
    parser = get_parser()
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=default_prompt_path,
        help="Path to prompt template file",
    )
    args = parser.parse_args()

    DATASET = args.dataset
    SPLIT = args.split
    LLM_CONFIG = args.llm_config
    EVAL_OUTPUT_DIR = args.eval_output_dir
    MAX_ITERATIONS = args.max_iterations
    EVAL_N_LIMIT = args.eval_n_limit
    EVAL_NOTE = args.eval_note
    PROMPT_PATH = args.prompt_path

    # Create LLM instance
    # api_key = os.getenv("LITELLM_API_KEY")
    # if not api_key:
    #     raise ValueError("LITELLM_API_KEY environment variable is not set")
    llm_config = get_llm_config_arg(LLM_CONFIG)
    llm = LLM(
        model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p,
        top_k=llm_config.top_k,
        # native_tool_calling=True,
    )
    # llm = LLM(
    #     model="openai/Qwen/Qwen3-8B",
    #     api_key="aditya_vllm",
    #     base_url="http://127.0.0.1:8000/v1",
    #     temperature=0.6,
    #     top_p=0.95,
    #     top_k=20,
    # )

    dataset_description = DATASET.replace("/", "__") + "-" + SPLIT.replace("/", "__")

    # Construct proper structured output directory path
    structured_output_dir = construct_eval_output_dir(
        base_dir=EVAL_OUTPUT_DIR,
        dataset_name=dataset_description,
        model=llm.model,
        max_iterations=MAX_ITERATIONS,
        eval_note=EVAL_NOTE,
    )

    metadata = make_metadata(
        llm,
        dataset_description,
        MAX_ITERATIONS,
        structured_output_dir,
        details={},
        dataset=DATASET,
        data_split=SPLIT,
        prompt_path=PROMPT_PATH,
        eval_n_limit=EVAL_N_LIMIT,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
    )

    # Check RUNTIME environment variable to determine mode
    runtime_mode = os.getenv("RUNTIME", "local").lower()
    
    if runtime_mode == "remote":
        run_remote_evaluation(llm, metadata, args.eval_num_workers)
    else:
        run_local_mode(args, llm, metadata)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
