import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
import json
import spacy
import math
import nltk
import numpy as np
from nltk import tokenize
from nltk.corpus import stopwords
from pydantic import BaseModel
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from operator import itemgetter
from pysentimiento import create_analyzer

stop_words = set(stopwords.words('english'))
settings = get_settings()


@Language.factory("custom_language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()


nlp_en = spacy.load("en_core_web_sm")
nlp_en.add_pipe("custom_language_detector", name="language_detector", last=True)

emotion_analyzer = create_analyzer(task='emotion', lang='en')


def check_sent(word, sentences):
    final = [all([w in x for w in word]) for x in sentences]
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))


def get_metadata(text):
    doc = nlp_en(text)

    return doc._.language, emotion_analyzer.predict(text)


def get_text_tf_idf_score(text: str):
    # get total words in text
    total_words = text.split()

    # remove stop words
    stop_words_list = set(stopwords.words('english'))
    total_words = [w for w in total_words if w not in stop_words_list]

    # remove punctuation
    total_words = [w for w in total_words if w.isalpha()]

    # count total words in lyrics
    total_words_len = len(total_words)

    # count total sentences in lyrics
    total_sentences = tokenize.sent_tokenize(text)
    total_sent_len = len(total_sentences)

    # calculate tf for each word
    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1

    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y / int(total_words_len)) for x, y in tf_score.items())

    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len) / y)) for x, y in idf_score.items())
    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}

    return tf_idf_score


def filter_insignificant(chunk, tag_suffixes):
    return [(word, tag) for word, tag in chunk if
            all(suffix not in tag and not tag.endswith(suffix) for suffix in tag_suffixes)]


def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key=itemgetter(1), reverse=True))
    # for each word get nltk tag
    result = nltk.pos_tag(result)
    tags = ['DT', 'CC', 'PRP', 'IN', 'RB', 'VBG', 'VBZ']
    result = filter_insignificant(result, tags)
    result = [word.lower() for word, tag in result]
    return result[:n]


class MyService(Service):
    """
    Sentiment analysis service
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Sentiment Analysis",
            slug="sentiment-analysis",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="text", type=[FieldDescriptionType.TEXT_PLAIN]),
            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.APPLICATION_JSON]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.SENTIMENT_ANALYSIS,
                    acronym=ExecutionUnitTagAcronym.SENTIMENT_ANALYSIS,
                ),
            ],
            has_ai=True,
            # OPTIONAL: CHANGE THE DOCS URL TO YOUR SERVICE'S DOCS
            docs_url="https://docs.swiss-ai-center.ch/reference/core-concepts/service/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        # Get the text to analyze from storage
        text = data["text"].data
        # Convert bytes to string
        text = text.decode("utf-8")
        # Get the language and sentiments
        language, sentiments = get_metadata(text)
        top_words = get_top_n(get_text_tf_idf_score(text), 10)

        # https://stackoverflow.com/a/57915246
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        json_result = {
            "language": language["language"],
            "sentiments": sentiments.probas,
            "top_words": top_words,
        }

        return {
            "result": TaskData(
                data=json.dumps(json_result, cls=NpEncoder, ensure_ascii=False).encode("utf-8"),
                type=FieldDescriptionType.APPLICATION_JSON
            )
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


api_summary = """
Analyze sentiment and emotions of a given text.
"""

api_description = """
Analyze sentiment of a given text. Returns a JSON object with the following fields:
- `language`: the language of the text
- `sentiment`: the sentiments detected in the text
- `top_words`: the top words in the text
"""

# Define the FastAPI application with information
# TODO: 7. CHANGE THE API TITLE, VERSION, CONTACT AND LICENSE
app = FastAPI(
    lifespan=lifespan,
    title="Sentiment Analysis API.",
    description=api_description,
    version="0.0.1",
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)

class Data(BaseModel):
    text: str

@app.post("/process", tags=['Process'])
def handle_process(data: Data):
    result = MyService().process({"text": TaskData(data=data.text, type=FieldDescriptionType.TEXT_PLAIN)})

    data = json.loads(result["result"].data)
    return data
