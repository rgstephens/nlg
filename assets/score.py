# Rasa Technologies GmbH
# Copyright 2021 Rasa Technologies GmbH
import argparse
import os
from typing import Text, Any, Dict, Optional, Tuple, Union
import shutil
from subprocess import call, STDOUT
import requests
import logging
import questionary
from urllib.parse import urljoin
import datetime
from base64 import b64encode
from pathlib import Path
from itertools import groupby

from rasa.shared.utils.cli import print_success, print_error_and_exit, print_info
from rasa.shared.utils.io import read_yaml, read_yaml_file, write_yaml
from rasa.shared.data import get_data_files
# from rasa.nlu.config import load as load_nlu_config
from rasa.shared.core.constants import DEFAULT_INTENTS
import rasa.cli.utils
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.constants import (
    CONFIG_SCHEMA_FILE,
    DEFAULT_E2E_TESTS_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_RESULTS_PATH,
)
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.core.training_data.structures import RuleStep, StoryStep

# ===========================================#
# These are internal settings that probably  #
# shouldn't be changed                       #
# -------------------------------------------#
RASA_SCORECARD_URL = "https://www.enterprise-scorecard.rasa.com/v1/"
# -------------------------------------------#

DEFAULT_SCORECARD = """
version: 1.0
scorecard:
  manual:
    CI:
      ci_runs_data_validation: false
      ci_trains_model: false
      ci_runs_rasa_test: false
      test_dir: tests
      ci_builds_action_server: false
    CD:
      ci_deploys_action_server: false
      infrastructure_as_code: false
      has_test_environment: false
      ci_runs_vulnerability_scans: false
    training_data_health:
      connected_channels: []
    success_kpis:
      automated_conversation_tags: []
  auto:
    CI:
      code_and_training_data_in_git: false
      rasa_test_coverage: 0.0
    training_data_health:
      num_min_example_warnings: null
      num_confused_intent_warnings: null
      num_confidence_warnings: null
      num_wrong_annotation_warnings: null
      num_precision_recall_warnings: null
      num_real_conversations: 0
      num_annotated_messages: 0
    success_kpis:
      has_fallback: false
      num_reviewed_conversations: 0
      num_tagged_conversations: 0
    training_data:
      num_intents: 0
      num_entities: 0
      num_actions: 0
      num_slots: 0
      num_responses: 0
      num_forms: 0
      num_retrieval_intents: 0
      num_stories: 0
      num_rules: 0
      avg_story_len: 0
      avg_rule_len: 0
      avg_entity_examples: 0
      avg_intent_examples: 0

"""

DataType = Dict[Text, Any]


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect data for scorecard.")
    parser.add_argument(
        "--username",
        "--user",
        help=(
            "Username for Rasa Enterprise. Can be set with the environment variable"
            " `RASA_ENTERPRISE_USER`"
        ),
        default=os.environ.get("RASA_ENTERPRISE_USER"),
    )
    parser.add_argument(
        "--password",
        "--pass",
        help=(
            "Password for Rasa Enterprise. Can be set with the environment variable"
            " `RASA_ENTERPRISE_PASSWORD`"
        ),
        default=os.environ.get("RASA_ENTERPRISE_PASSWORD"),
    )
    parser.add_argument(
        "--url",
        help=(
            "Host URL for Rasa Enterprise. Can be set with the environment variable"
            " `RASA_ENTERPRISE_URL`"
        ),
        default=os.environ.get("RASA_ENTERPRISE_URL"),
    )
    parser.add_argument(
        "--file",
        required=False,
        help="File containing your scores. Will be overwritten!",
    )
    parser.add_argument(
        "--debug", required=False, action="store_true", help="Turn on verbose logging."
    )
    parser.add_argument(
        "--skip-wizard",
        required=False,
        action="store_true",
        help="Skip running the wizard. User will not be prompted.",
    )
    parser.add_argument(
        "--skip-api",
        required=False,
        action="store_true",
        help=(
            "Skip fetching data from the Rasa Enterprise API. Useful if you haven't"
            " deployed Rasa Enterprise yet."
        ),
    )
    parser.add_argument(
        "--on-the-fly-insights",
        required=False,
        nargs="?",
        const="results",
        help=(
            "Calculate NLU insights on-the-fly based on cross-validation results."
            " Will skip fetching NLU Insights from the Rasa Enterprise API."
            " Accepts an optional path to results, defaults to  ./results."
        ),
    )
    parser.add_argument(
        "view",
        nargs="?",
        help=(
            "Load an existing scores file to generate a shareable link"
            " and display scorecard in the browser."
            " Skips fetching data or calculating scores."
        ),
    )

    return parser


def read_existing_scores_from_file(filename: str) -> DataType:
    try:
        data = read_yaml_file(filename)
    except:  # noqa: E722
        print_error_and_exit(f"Could not read file {filename} as yaml.")
    return data


def _create_backup_of_file(filename: str) -> str:
    date = datetime.date.today().strftime("%Y%m%d")
    backup_name = f"backup_{date}_{filename}"
    shutil.copy(filename, backup_name)
    return backup_name


def _is_cwd_git_dir() -> bool:
    logging.debug("Checking there is a git repo in this directory.")
    try:
        return call(["git", "branch"], stderr=STDOUT, stdout=open(os.devnull, "w")) == 0
    except:  # noqa: E722
        git_path = Path(__file__).parent.resolve() / "./.git"
        logging.debug(f"git command not available. Checking existence of: {git_path}")
        return git_path.exists()


def _has_fallback() -> bool:
    logging.debug("Checking if there is a fallback classifier in NLU pipeline.")

    try:
        config = str(
            rasa.cli.utils.get_validated_path("config.yml", "config", DEFAULT_CONFIG_PATH)
        )
        config_importer = TrainingDataImporter.load_from_dict(config_path=config)

        config_dict = config_importer.get_config()
        # config = load_nlu_config("./config.yml")
        logging.debug(f"NLU config has components: {config_dict['pipeline']}.")
        # return "FallbackClassifier" in config_dict.component_names
        contains_fallback = any(d.get('name') == 'Fallback' for d in config_dict['pipeline'])
        return contains_fallback
    except:  # noqa: E722
        logging.error("could not load config from file: config.yml")
        return False


def update_scores(
    data: DataType,
    token: Optional[str],
    url: Optional[str],
    crossval_results: Optional[str],
) -> DataType:
    data = update_data_from_repo(data)
    if token and url:
        data = update_data_from_rasa_enterprise(data, token, url, crossval_results)
    elif crossval_results:
        warnings = count_warnings(url, token, crossval_results)
        data["scorecard"]["auto"]["training_data_health"].update(warnings)

    return data


def get_auth_token(
    user: Optional[str], password: Optional[str], url: Optional[str]
) -> Optional[str]:
    if not user or not password or not url:
        print_error_and_exit(
            "Rasa Enterprise login credentials not set. Run the script with `--help`"
            " for more information."
        )

    url = urljoin(url, "/api/auth")
    payload = {"username": user, "password": password}
    response = requests.post(url, json=payload)
    try:
        token = response.json()["access_token"]  # type: str
        return token
    except:  # noqa: E722
        logging.error(
            "Unable to authenticate against Rasa Enterprise API. Error"
            f" {response.status_code}"
        )
        logging.error(response.text)
        return None


def count_tagged_conversations(url: str, token: str, tags: bool = None) -> int:
    logging.debug("Counting conversations with tags.")
    url = urljoin(url, "/api/data_tags")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    total = 0
    try:
        data_tags = response.json()
    except:  # noqa: E722
        logging.error(
            "Error calling Rasa Enterprise API to count tagged conversations."
        )
        logging.error(response)
        return total

    for tag in data_tags:
        if tag["value"] in tags:
            num_conversations = len(tag["conversations"])
            logging.debug(
                f"Found {num_conversations} conversations tagged with {tag['value']}."
            )
            total += num_conversations
    return total


def count_conversations(
    url: str, token: str, unread: bool = False, rasa_channel: bool = False
) -> int:
    params = {"limit": 1, "exclude_self": True}
    if unread:
        params["review_status"] = "unread"
    if rasa_channel:
        params["input_channels"] = "rasa"

    logging.debug(f"Counting conversations with params: {params}.")
    url = urljoin(url, "/api/conversations")
    headers = {"Authorization": f"Bearer {token}"}
    response = None
    try:
        response = requests.get(url, headers=headers, params=params)
        num_conversations = int(response.headers.get("X-Total-Count", 0))
        return num_conversations
    except:  # noqa: E722
        logging.error("Error calling Rasa Enterprise API to count conversations.")
        logging.error(response)
        return 0


def count_messages(url: str, token: str, exclude_training_data: bool = False) -> int:
    params = {"limit": 1, "exclude_training_data": exclude_training_data}
    url = urljoin(url, "/api/projects/default/logs")
    headers = {"Authorization": f"Bearer {token}"}
    response = None
    try:
        response = requests.get(url, headers=headers, params=params)
        num_messages = int(response.headers.get("X-Total-Count", 0))
        return num_messages
    except:  # noqa: E722
        logging.error("Error calling Rasa Enterprise API to count messages.")
        logging.error(response)
        return 0


insight_key_map = {
    "MinimumExampleInsightCalculator": "num_min_example_warnings",
    "ConfusionInsightCalculator": "num_confused_intent_warnings",
    "NLUInboxConfidenceInsightCalculator": "num_confidence_warnings",
    "WrongAnnotationInsightCalculator": "num_wrong_annotation_warnings",
    "ClassBiasInsightCalculator": "num_precision_recall_warnings",
}


def calculate_insights_for_ci(
    linter_config: DataType = None, test_results: str = str(Path("results"))
) -> DataType:
    """Calculates NLU insights based on local cross-validation results"""
    import asyncio
    import contextlib

    from rasax.community.services.insights.insight_calculator import (
        _get_insights_for_cli,
    )

    context_manager = contextlib.nullcontext(None)

    with context_manager as session:
        print_info(
            "Calculating intent insights based on local cross-validation results ... 🔎"
        )
        loop = asyncio.get_event_loop()
        warnings = loop.run_until_complete(
            _get_insights_for_cli(linter_config, test_results, session,)
        )

    if warnings:
        logging.debug("Suggestions to improve training data were found 🚨")
    else:
        logging.debug("No suggestions to improve training data were found")

    return warnings


def count_warnings(url: str, token: str, crossval_results: Optional[str]) -> DataType:
    result = {
        "num_min_example_warnings": None,
        "num_confused_intent_warnings": None,
        "num_confidence_warnings": None,
        "num_wrong_annotation_warnings": None,
        "num_precision_recall_warnings": None,
    }

    headers = {"Authorization": f"Bearer {token}"}

    url = urljoin(url, "/api/insights/nlu")
    version_url = urljoin(url, "/api/version")
    try:
        if crossval_results:
            calculated_warnings = calculate_insights_for_ci(
                test_results=crossval_results
            )
            warnings = [
                {
                    "source": insight.source,
                    "details": insight.details,
                    "description": insight.description,
                }
                for intent in calculated_warnings.values()
                for insight in intent
            ]
            if warnings:
                logging.debug(
                    "NLU insight reports present, replacing default 'null' warning"
                    " counts with 0"
                )
                result = dict.fromkeys(result, 0)

        else:
            version_response = requests.get(version_url, headers=headers)
            version = version_response.json().get("rasa-x")
            if version < "0.36.0":
                logging.warning(
                    "Rasa X version is < 0.36 so cannot retrieve NLU Insights"
                )
                return dict.fromkeys(result, None)

            response = requests.get(url, headers=headers)

            if response.status_code == 404:
                logging.warning("Failed to retrieve NLU Insights from Rasa X API")
                return dict.fromkeys(result, None)

            insight_reports = response.json()

            latest_report = next(
                (
                    report
                    for report in reversed(insight_reports)
                    if report["status"] == "success"
                ),
                None,
            )

            if latest_report is None:
                logging.error(
                    "No successful NLU insight reports found on Rasa Enterprise API"
                )
                return dict.fromkeys(result, None)
            else:
                logging.debug(
                    "NLU insight reports present, replacing default 'null' warning"
                    " counts with 0"
                )
                result = dict.fromkeys(result, 0)

            url = urljoin(url, f"nlu/{latest_report['id']}")
            response = requests.get(url, headers=headers)
            report = response.json()

            warnings = [
                warnings
                for intent in report["intent_evaluation_results"]
                for warnings in intent["intent_insights"]
            ]
            warnings.sort(key=lambda w: w["source"])

        for warning_type, group in groupby(warnings, lambda w: w["source"]):
            key = insight_key_map.get(warning_type)
            if key:
                result[key] = len(list(group))

        return result
    except:  # noqa: E722
        logging.error("Error processing NLU insight report")
        return dict.fromkeys(result, None)


def load_domain(domain_path: Path = Path(".")):
    from rasa.shared.core.domain import Domain

    domain_roots = []

    if domain_path.is_file():
        domain_roots = [domain_path]
    else:
        # Ignore the data folder because responses are considered valid domain files
        # Really we need to fix `rasa` to better detect these files so we can just throw
        # a directory to Domain.from_path() to walk

        domain_roots = []
        for p in domain_path.iterdir():
            if not p.name.startswith(".") and p.is_file() and p.name != "data":
                try:
                    if Domain.is_domain_file(p):
                        domain_roots.append(p)
                except UnicodeDecodeError:
                    # Rasa sometimes throws a UnicodeDecodeError
                    # just skip these yml files as candidate domain files
                    # possible causes:
                    # (-) issue 102: there is a `\` in the yml file
                    pass

    domain = Domain.empty()

    for d in domain_roots:
        domain = domain.merge(Domain.from_path(d))

    return domain


def calculate_training_data(data: DataType) -> float:
    logging.debug(
        "Calculating training data totals."
    )
    result = {
        # domain
        "num_intents": 0,
        "num_entities": 0,
        "num_actions": 0,
        "num_slots": 0,
        "num_responses": 0,
        "num_forms": 0,
        "num_retrieval_intents": 0,
        "num_stories": 0,
        "num_rules": 0,
        "avg_story_len": 0,
        "avg_rule_len": 0,
        "avg_entity_examples": 0,
        "avg_intent_examples": 0,
    }
    config = str(
        rasa.cli.utils.get_validated_path("config.yml", "config", DEFAULT_CONFIG_PATH)
    )
    domain = rasa.cli.utils.get_validated_path(
        "domain.yml", "domain", DEFAULT_DOMAIN_PATH, none_is_valid=False
    )
    data_path = rasa.cli.utils.get_validated_path("data", "nlu", DEFAULT_DATA_PATH)
    # data_path = rasa.shared.data.get_nlu_directory(data_path)
    training_files = rasa.shared.data.get_data_files(
        data_path, YAMLStoryReader.is_stories_file
    )
    training_files += rasa.shared.data.get_data_files(
        data_path, rasa.shared.data.is_nlu_file
    )
    # config_importer = TrainingDataImporter.load_from_dict(config_path=config)
    # Args:
    #     config: Path to the config file.
    #     domain: Path to the domain file.
    #     training_files: List of paths to training data files.
    file_importer = TrainingDataImporter.load_from_config(
        domain_path=domain, training_data_paths=training_files, config_path=config
    )

    # Process domain
    domain = file_importer.get_domain()
    intents = {i: 0 for i in domain.intents if i not in DEFAULT_INTENTS}
    result["num_intents"] = len(intents.keys())
    entities = {i: 0 for i in domain.entities}
    result["num_entities"] = len(entities.keys())
    actions = {i: 0 for i in domain.user_actions}
    result["num_actions"] = len(actions.keys())
    result["num_slots"] = len(domain.slots) - 1
    result["num_responses"] = len(domain.responses.keys())
    result["num_forms"] = len(domain.forms.keys())
    result["num_retrieval_intents"] = len(domain.retrieval_intents)

    # get's stories & rule
    stories = file_importer.get_stories()
    num_rule_steps = 0
    num_story_steps = 0
    for story in stories.story_steps:
        if isinstance(story, RuleStep):
            result['num_rules'] += 1
            events = story.get_rules_events()
            for evt in events:
                if hasattr(evt, 'intent') or (hasattr(evt, 'action_name') and evt.action_name != '...') or (hasattr(evt, 'type_name') and evt.type_name == 'active_loop') or evt.type_name == 'slot':
                    num_rule_steps += 1
        elif isinstance(story, StoryStep):
            result['num_stories'] += 1
            for evt in story.events:
                if hasattr(evt, 'intent') or (hasattr(evt, 'action_name') and evt.action_name != '...'):
                    num_story_steps += 1

    if result['num_rules']:
        result['avg_rule_len'] = round(num_rule_steps / result['num_rules'], 1)
    if result['num_stories']:
        result['avg_story_len'] = round(num_story_steps / result['num_stories'], 1)

    # nlu training data
    nlu = file_importer.get_nlu_data()
    if nlu.number_of_examples_per_intent:
        result['avg_intent_examples'] = round(sum(nlu.number_of_examples_per_intent.values()) / len(nlu.number_of_examples_per_intent.values()), 1)
    if nlu.number_of_examples_per_entity:
        result['avg_entity_examples'] = round(sum(nlu.number_of_examples_per_entity.values()) / len(nlu.number_of_examples_per_entity.values()), 1)
    return result

def calculate_test_coverage(data: DataType) -> float:
    logging.debug(
        "Calculating test coverage by inspecting domain and test conversations."
    )
    result = {
        "num_intents": 0,
        "num_entities": 0,
        "num_actions": 0,
        "num_slots": 0,
    }
    domain = load_domain()

    intents = {i: 0 for i in domain.intents if i not in DEFAULT_INTENTS}
    result["num_intents"] = len(intents.keys())
    logging.debug(f"Found {len(intents)} intents in domain.")
    entities = {i: 0 for i in domain.entities}
    result["num_entities"] = len(entities.keys())
    logging.debug(f"Found {len(entities)} entities in domain.")
    actions = {i: 0 for i in domain.user_actions}
    result["num_actions"] = len(actions.keys())
    logging.debug(f"Found {len(actions)} actions in domain.")
    result["num_slots"] = len(domain.slots) - 1
    logging.debug(f"Found {len(actions)} actions in domain.")

    data["scorecard"]["auto"]["training_data"].update(result)

    try:
        user_slots = domain._user_slots
    except AttributeError:
        user_slots = domain.slots

    slots = {i.name: 0 for i in user_slots}
    logging.debug(f"Found {len(slots)} slots in domain.")

    tests_path = data["scorecard"]["manual"]["CI"]["test_dir"]
    tmp_dir = get_data_files(tests_path, YAMLStoryReader.is_test_stories_file)
    if (tmp_dir):
        for test_file in tmp_dir:
        # for test_file in os.listdir(tests_path):
            with open(test_file, "r") as tempfile:
            # with open(os.path.join(tmp_dir, test_file), "r") as tempfile:
                for line in tempfile:
                    for intent in intents.keys():
                        if intent in line:
                            intents[intent] += 1
                    for entity in entities.keys():
                        if entity in line:
                            entities[entity] += 1
                    for action in actions.keys():
                        if action in line:
                            actions[action] += 1
                    for slot in slots.keys():
                        if slot in line:
                            slots[slot] += 1

    logging.debug(f"Appearances per intent in test conversations: {intents}.")
    covered_intents = [i for i, tot in intents.items() if tot > 1]
    covered_entities = [i for i, tot in entities.items() if tot > 1]
    covered_actions = [i for i, tot in actions.items() if tot > 1]
    covered_slots = [i for i, tot in slots.items() if tot > 1]
    logging.debug(
        f"{len(covered_intents)}/{len(intents)} intents appear in at least 2 test"
        " conversations."
    )
    logging.debug(
        f"{len(covered_entities)}/{len(entities)} entities appear in at least 2 test"
        " conversations."
    )
    logging.debug(
        f"{len(covered_actions)}/{len(actions)} actions appear in at least 2 test"
        " conversations."
    )
    logging.debug(
        f"{len(covered_slots)}/{len(slots)} slots appear in at least 2 test"
        " conversations."
    )
    numerator = (
        len(covered_intents)
        + len(covered_entities)
        + len(covered_actions)
        + len(covered_slots)
    )
    denominator = len(intents) + len(entities) + len(actions) + len(slots)

    if denominator == 0:
        return 0

    return round(100 * numerator / denominator, 1)


def update_data_from_repo(data: DataType) -> DataType:
    """Inspect repo and determine if necessary steps are in place."""
    logging.debug("Inspecting contents of current directory.")

    data["scorecard"]["auto"]["CI"]["code_and_training_data_in_git"] = _is_cwd_git_dir()
    data["scorecard"]["auto"]["success_kpis"]["has_fallback"] = _has_fallback()

    data["scorecard"]["auto"]["CI"]["rasa_test_coverage"] = calculate_test_coverage(
        data
    )

    data["scorecard"]["auto"]["training_data"] = calculate_training_data(
        data
    )

    return data


def update_data_from_rasa_enterprise(
    data: DataType, token: str, url: str, crossval_results: Optional[str]
) -> DataType:
    logging.debug("Fetching data from Rasa Enterprise API.")

    if url is None or token is None:
        return data

    logging.debug("Successfully retrieved an auth token.")

    total_conversations = count_conversations(url, token)
    real_conversations = total_conversations - count_conversations(
        url, token, rasa_channel=True
    )
    reviewed_conversations = total_conversations - count_conversations(
        url, token, unread=True
    )
    automated_conversation_tags = data["scorecard"]["manual"]["success_kpis"][
        "automated_conversation_tags"
    ]
    tagged_conversations = 0
    if automated_conversation_tags:
        logging.debug(
            f"Counting conversations tagged with one of {automated_conversation_tags}."
        )
        tagged_conversations = count_tagged_conversations(
            url, token, tags=automated_conversation_tags
        )

    data["scorecard"]["auto"]["training_data_health"][
        "num_real_conversations"
    ] = real_conversations
    data["scorecard"]["auto"]["success_kpis"][
        "num_reviewed_conversations"
    ] = reviewed_conversations
    data["scorecard"]["auto"]["success_kpis"][
        "num_tagged_conversations"
    ] = tagged_conversations

    raw_num_messages = count_messages(url, token, exclude_training_data=False)
    messages_not_in_training_data = count_messages(
        url, token, exclude_training_data=True
    )
    num_annotated_messages = raw_num_messages - messages_not_in_training_data

    data["scorecard"]["auto"]["training_data_health"][
        "num_annotated_messages"
    ] = num_annotated_messages

    warnings = count_warnings(url, token, crossval_results)

    data["scorecard"]["auto"]["training_data_health"].update(warnings)

    return data


def fill_boolean_question(
    data: DataType, section: str, key: str, question: str
) -> DataType:
    answer = questionary.confirm(question).ask()
    data["scorecard"]["manual"][section][key] = answer
    return data


def run_auth_wizard(
    user: Optional[str], password: Optional[str], url: Optional[str]
) -> Tuple[Union[Any, str], ...]:
    user = (
        questionary.text("What is your Rasa Enterprise username?").ask()
        if user is None
        else user
    )
    password = (
        questionary.password("What is your Rasa Enterprise password?").ask()
        if password is None
        else password
    )
    url = (
        questionary.text("What is the URL of your Rasa Enterprise instance?").ask()
        if url is None
        else url
    )

    return user, password, url


def run_wizard() -> DataType:
    should_create_new_file = questionary.confirm(
        "Do you want to start a new scorecard from scratch?"
    ).ask()
    if not should_create_new_file:
        print_success(
            "OK. Stopping the script. Please provide an existing score yaml file"
            " through the `--file` argument"
        )
        exit(0)

    print_success(
        "Great! I'm going to ask a few questions to fill in your initial scorecard. The"
        " scores evaluate the health of your training data, how well you are measuring"
        " the success of your assistant, and the maturity of your CI/CD pipeline. This"
        " is partly automated, but we'll have to ask some questions about your CI/CD"
        " setup. You can always go back and update your answers by editing your"
        " scorecard file."
    )
    data = read_yaml(DEFAULT_SCORECARD)

    channels = ["website", "messenger", "twilio", "whatsapp", "custom"]
    messaging_channels = questionary.checkbox(
        "Please use the SPACE KEY to select the channels where your assistant is"
        " available.",
        choices=channels,
    ).ask()
    data["scorecard"]["manual"]["training_data_health"][
        "connected_channels"
    ] = messaging_channels

    questions = {
        "ci_runs_data_validation": (
            "Does your CI server validate your data using `rasa data validate`?"
        ),
        "ci_trains_model": (
            "Does your CI server train a model for testing? E.g. with `rasa train`?"
        ),
        "ci_runs_rasa_test": (
            "Does your CI server run through test conversations? E.g. with `rasa test`?"
        ),
        "ci_builds_action_server": (
            "Does your CI server build an image for your custom action server?"
        ),
    }
    for key, question in questions.items():
        data = fill_boolean_question(data, "CI", key, question)

    test_dir = questionary.text(
        "Please pick the directory where your test stories are saved.", default="tests"
    ).ask()
    if os.path.isdir(test_dir):
        data["scorecard"]["manual"]["CI"]["test_dir"] = test_dir
    else:
        logging.error(f"{test_dir} does not seem to be a valid directory. Skipping.")

    questions = {
        "ci_deploys_action_server": (
            "Does your CI server automatically deploy your custom action server?"
        ),
        "infrastructure_as_code": (
            "Do you have all of your infrastructure configured as code? E.g. with"
            " terraform."
        ),
        "has_test_environment": (
            "Do you have a test or staging environment to try changes before they go to"
            " production?"
        ),
        "ci_runs_vulnerability_scans": (
            "Does your CI server run vulnerability scans on all images?"
        ),
    }
    for key, question in questions.items():
        data = fill_boolean_question(data, "CD", key, question)

    auto_tags = ""

    try:
        auto_tags = questionary.text(
            "If you are using the Rasa Enterprise API to tag conversations"
            " programatically, please enter the names of the relevant tags here (comma"
            " separated)"
        ).ask()
    except:  # noqa: E722
        print_error_and_exit(
            "Your questionary version is not up to date"
            " please update it by running `pip install -U questionary`"
        )

    if len(auto_tags) > 0:
        auto_tags = [s.strip() for s in auto_tags.split(",")]
        data["scorecard"]["manual"]["success_kpis"][
            "automated_conversation_tags"
        ] = auto_tags

    return data


def prompt_web_ui(url: str) -> None:
    should_open_url = questionary.confirm(
        "Would you like to open up your results in the web-based interface?"
    ).ask()
    if should_open_url:
        import webbrowser

        webbrowser.open(url)


def get_shareable_link(yaml: str) -> str:
    encoded = b64encode(yaml.encode("utf-8"))
    string = encoded.decode("utf-8").replace("+", "-").replace("/", "_").rstrip("=")

    url = urljoin(RASA_SCORECARD_URL, f"/?scorecard={string}")
    return url


def generate_yaml(data) -> str:
    from io import StringIO

    schema_url = urljoin(RASA_SCORECARD_URL, "schema.json")
    buffer = StringIO("")
    write_yaml(data, buffer, should_preserve_key_order=True)
    yaml = f"# yaml-language-server: $schema={schema_url}\n{buffer.getvalue()}"
    return yaml


def load_scorecard(filename: str = None) -> str:
    if not filename:
        data = read_yaml(DEFAULT_SCORECARD)
        logging.info("No scores file provided, using default values")
    else:
        data = read_existing_scores_from_file(filename)
    yaml = generate_yaml(data)
    return yaml


def update_scorecard(
    skip_api: bool,
    skip_wizard: bool,
    filename: str,
    user: str,
    password: str,
    url: str,
    crossval_results: str,
):
    token = None
    if not skip_api:
        user, password, url = run_auth_wizard(user=user, password=password, url=url)
        token = get_auth_token(user, password, url)

    if filename:
        backup_filename = _create_backup_of_file(filename)
        logging.info(f"backed up {filename} to {backup_filename}")
        data = read_existing_scores_from_file(filename)

    elif not skip_wizard:
        data = run_wizard()
        filename = "scores.yml"
        logging.info(
            f"Wizard complete. Creating a new scorecard file called {filename}"
        )

    else:
        data = read_yaml(DEFAULT_SCORECARD)
        filename = "scores.yml"
        logging.info("Skipping wizard and taking default values.")

    logging.info(
        "updating scores based on current directory and data from the Rasa"
        " Enterprise API"
    )

    data = update_scores(data, token, url, crossval_results)
    yaml = generate_yaml(data)
    with open(filename, "w") as fd:
        fd.write(yaml)

    print_success(f"Your scores have been updated and saved to {filename}.")
    return yaml


def main() -> None:
    arg_parser = _create_argument_parser()
    cmdline_arguments = arg_parser.parse_args()
    view = cmdline_arguments.view
    filename = cmdline_arguments.file
    user = cmdline_arguments.username
    password = cmdline_arguments.password
    url = cmdline_arguments.url
    skip_api = cmdline_arguments.skip_api
    skip_wizard = cmdline_arguments.skip_wizard
    crossval_results = cmdline_arguments.on_the_fly_insights

    log_level = logging.DEBUG if cmdline_arguments.debug else logging.INFO
    logging.basicConfig(level=log_level)

    if view:
        yaml = load_scorecard(filename)

    else:
        yaml = update_scorecard(
            skip_api, skip_wizard, filename, user, password, url, crossval_results
        )

    url = get_shareable_link(yaml)
    print_info(
        f"You can view your scorecard at:\n{url}"
        "\nCopy the link above to share a view of your scorecard with others."
    )

    if view or not skip_wizard:
        prompt_web_ui(url)


if __name__ == "__main__":
    main()