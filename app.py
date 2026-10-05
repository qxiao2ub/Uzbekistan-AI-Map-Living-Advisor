from __future__ import annotations

import base64
import html
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import quote

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "cities_uzbekistan.csv"
ASSET_DIR = APP_DIR / "assets"
FEEDBACK_PATH = APP_DIR / "feedback_state.json"

FEATURE_COLUMNS = [
    "affordability",
    "grocery_affordability",
    "entertainment",
    "sunset_scenery",
    "safety",
    "jobs",
    "internet",
    "healthcare",
    "education",
    "heritage",
    "nature",
    "climate_comfort",
    "quietness",
    "mobility",
]

FEATURE_LABELS = {
    "affordability": "Affordable living",
    "grocery_affordability": "Affordable groceries",
    "entertainment": "Entertainment",
    "sunset_scenery": "Sunset & scenery",
    "safety": "Safety",
    "jobs": "Jobs & career",
    "internet": "Internet & remote work",
    "healthcare": "Healthcare",
    "education": "Education",
    "heritage": "History & culture",
    "nature": "Nature access",
    "climate_comfort": "Climate comfort",
    "quietness": "Quiet lifestyle",
    "mobility": "Transportation",
}

SYNONYMS = {
    "affordability": [
        "cheap", "cheapest", "affordable", "low cost", "low-cost", "budget",
        "inexpensive", "save money", "low rent", "reasonable rent",
    ],
    "grocery_affordability": [
        "cheap grocery", "cheap groceries", "grocery price", "food price",
        "affordable food", "low food cost", "market price", "cheap market",
    ],
    "entertainment": [
        "entertainment", "nightlife", "fun", "activities", "events", "concert",
        "cinema", "shopping", "restaurants", "social life", "things to do",
    ],
    "sunset_scenery": [
        "sunset", "sunsets", "scenic", "beautiful view", "view", "photography",
        "sky", "romantic", "landscape",
    ],
    "safety": ["safe", "safety", "secure", "low crime", "family friendly"],
    "jobs": ["job", "jobs", "career", "employment", "business", "salary", "startup"],
    "internet": [
        "internet", "wifi", "remote work", "digital nomad", "online work",
        "technology", "tech", "fast connection",
    ],
    "healthcare": ["hospital", "healthcare", "doctor", "medical", "clinic", "health"],
    "education": [
        "school", "education", "university", "college", "student", "children",
        "academic", "learning",
    ],
    "heritage": [
        "history", "historic", "heritage", "culture", "architecture", "museum",
        "old city", "silk road", "traditional",
    ],
    "nature": [
        "nature", "mountain", "mountains", "green", "park", "hiking", "outdoor",
        "river", "lake", "desert", "fresh air",
    ],
    "climate_comfort": [
        "comfortable weather", "mild climate", "climate", "weather", "not too hot",
        "not too cold", "pleasant weather",
    ],
    "quietness": ["quiet", "calm", "peaceful", "slow life", "less crowded", "relaxed"],
    "mobility": [
        "transport", "transportation", "metro", "bus", "airport", "walkable",
        "commute", "easy travel", "connected",
    ],
}

CITY_IMAGE_FILES = {
    'Tashkent': 'cities/tashkent.jpg',
    'Samarkand': 'cities/samarkand.jpg',
    'Bukhara': 'cities/bukhara.jpg',
    'Khiva': 'cities/khiva.jpg',
    'Nukus': 'cities/nukus.jpg',
    'Fergana': 'cities/fergana.jpg',
    'Andijan': 'cities/andijan.jpg',
    'Namangan': 'cities/namangan.jpg',
    'Qarshi': 'cities/qarshi.jpg',
    'Termez': 'cities/termez.jpg',
    'Jizzakh': 'cities/jizzakh.jpg',
    'Gulistan': 'cities/gulistan.jpg',
    'Navoi': 'cities/navoi.jpg',
    'Urgench': 'cities/urgench.jpg',
    'Kokand': 'cities/kokand.jpg',
    'Shahrisabz': 'cities/shahrisabz.jpg',
}
FALLBACK_CITY_IMAGE = "spot-forest.jpg"


# -----------------------------------------------------------------------------
# Localization
# -----------------------------------------------------------------------------
LANGUAGE_OPTIONS = {
    "English": "en",
    "Русский": "ru",
    "O'zbek": "uz",
}

TRANSLATIONS = {
    "en": {
        "nav_home": "Home", "nav_recommend": "Recommend", "nav_map": "Map & Compare",
        "nav_cities": "Cities", "nav_ai": "AI Lab", "nav_about": "About",
        "language": "Language", "app_visits": "Cumulative App Visits", "visit_note": "One visit is counted once per browser session.",
        "your_priorities": "Your priorities", "priority_caption": "Text is the main input. Sliders let you emphasize or correct specific priorities.",
        "num_recs": "Number of recommendations", "advanced": "Advanced preference sliders", "prototype_note": "Prototype note",
        "prototype_copy": "City scores are illustrative, not official statistics. Verify housing, employment, safety, healthcare, visa, and legal information before relocating.",
        "featured": "Featured matches", "featured_title": "Lifestyle ideas, ranked for you",
        "featured_sub": "A first look at cities selected by the same AI engine used in the full recommendation workspace.",
        "experience": "The experience", "experience_title": "Minimal design, useful intelligence",
        "experience_sub": "The uploaded UI's calm, image-led design language is preserved while the interaction is rebuilt around city discovery.",
        "advisor": "AI advisor", "advisor_title": "Describe your ideal place",
        "advisor_sub": "Use natural language now; detailed controls are available in the sidebar and the Recommend workspace.",
        "lifestyle_request": "Lifestyle request", "find_cities": "Find my cities", "updated": "Recommendations updated. Open the Recommend or Map & Compare workspace for full details.",
        "personalized": "Personalized advisor", "tell_ai": "Tell the AI what matters",
        "tell_ai_sub": "Text is the main signal. The ranking engine combines interpretable lifestyle features, NLP similarity, a neural model, and feedback.",
        "question": t("question"), "find_best": t("find_best"),
        "interpret": t("interpret"), "signals": t("signals"), "preference": "Preference", "weight": "Weight (%)", "top_matches": "Top matches",
        "why_fits": "Why it fits", "hybrid": "Hybrid components", "optional_live": "Optional live context", "city": t("city"),
        "fetch_context": t("fetch_context"), "temp": t("temp"), "feels": t("feels"), "conditions": t("conditions"), "source": t("source"),
        "map_compare": "Map & compare", "map_title": "See the recommendations geographically", "map_sub": "The map uses OpenStreetMap tiles and overlays the prototype Uzbekistan city profiles and current recommendation ranking.",
        "score_compare": "Recommendation score comparison", "choose_cities": t("choose_cities"), "profile": "Detailed city profile", "select_city": t("select_city"),
        "filter_cities": t("filter_cities"), "search_placeholder": "Search by city, region, or tag", "city_explorer": "City explorer", "city_explorer_title": "Browse every prototype profile",
        "city_explorer_sub": "Each profile now uses the city-specific photo supplied for that named city, while preserving the uploaded location-card design.",
        "region": "Region", "tags": "Tags", "score": "Prototype score", "dimension": "Dimension",
        "ai_lab": "AI laboratory", "ai_title": "How the recommendation brain works",
        "ai_sub": "The prototype intentionally combines explainable rules with machine learning, a neural-network demonstration, and feedback learning.",
        "data_archetypes": "Data-driven city archetypes", "team": "Project team",
        "production_warning": "For production, replace illustrative scores and synthetic training data with licensed, dated, auditable sources. The app itself does not use a database for visitor counting; the cumulative counter is maintained by an external counter service.",
        "about": "About the project", "about_title": "Living decisions, made easier to explore", "about_sub": "A portfolio-ready prototype that turns broad lifestyle preferences into a structured, visual city-comparison workflow.",
        "included": "What is included", "ui_preserve": "UI source preservation", "footer": "AI-assisted city discovery for Uzbekistan · Streamlit portfolio prototype",
        "counter_unavailable": "Counter temporarily unavailable",
        "like": "Like", "not_for_me": "Not for me", "feedback_saved": "Feedback saved for",
        "natural_language": "Natural-language understanding", "natural_desc": "Turn everyday phrases such as 'cheap groceries' or 'great sunsets' into measurable lifestyle priorities.",
        "map_first": "Map-first exploration", "map_first_desc": "See candidate cities geographically and inspect how recommendations relate across Uzbekistan.",
        "transparent": "Transparent scoring", "transparent_desc": "Combine preference features, text similarity, neural-network estimates, and feedback into an explainable score.",
        "feedback_learning": "Feedback learning", "feedback_desc": "Likes and dislikes update a lightweight bandit signal so the prototype can adapt over time.",
    },
    "ru": {
        "nav_home": "Главная", "nav_recommend": "Рекомендации", "nav_map": "Карта и сравнение", "nav_cities": "Города", "nav_ai": "ИИ-лаборатория", "nav_about": "О проекте",
        "language": "Язык", "app_visits": "Накопленные посещения", "visit_note": "Одно посещение учитывается один раз за сессию браузера.",
        "your_priorities": "Ваши приоритеты", "priority_caption": "Основной ввод — текст. Ползунки позволяют усилить или скорректировать отдельные приоритеты.",
        "num_recs": "Количество рекомендаций", "advanced": "Расширенные ползунки приоритетов", "prototype_note": "Примечание о прототипе",
        "prototype_copy": "Оценки городов являются демонстрационными, а не официальной статистикой. Перед переездом проверяйте жильё, работу, безопасность, медицину, визовые и юридические сведения.",
        "featured": "Избранные совпадения", "featured_title": "Варианты образа жизни, ранжированные для вас", "featured_sub": "Первые города, выбранные тем же ИИ-модулем, который используется в полном рабочем пространстве рекомендаций.",
        "experience": "Возможности", "experience_title": "Минималистичный дизайн и полезный интеллект", "experience_sub": "Спокойный визуальный стиль загруженного интерфейса сохранён, а взаимодействие перестроено вокруг поиска города.",
        "advisor": "ИИ-консультант", "advisor_title": "Опишите идеальное место", "advisor_sub": "Используйте обычный язык; дополнительные настройки находятся на боковой панели и на странице рекомендаций.",
        "lifestyle_request": "Запрос об образе жизни", "find_cities": "Найти мои города", "updated": "Рекомендации обновлены. Откройте раздел рекомендаций или карты для подробностей.",
        "personalized": "Персональный консультант", "tell_ai": "Расскажите ИИ, что важно", "tell_ai_sub": "Основной сигнал — текст. Ранжирование объединяет интерпретируемые параметры, NLP-сходство, нейросеть и обратную связь.",
        "question": "Какое место вы ищете?", "find_best": "Найти лучшие города", "interpret": "Как ИИ интерпретировал запрос", "signals": "Обнаруженные сигналы:", "preference": "Приоритет", "weight": "Вес (%)", "top_matches": "Лучшие совпадения",
        "why_fits": "Почему подходит", "hybrid": "Гибридные компоненты", "optional_live": "Дополнительный актуальный контекст", "city": "Город", "fetch_context": "Получить погоду и краткую сводку",
        "temp": "Температура", "feels": "По ощущениям", "conditions": "Условия", "source": "Открыть источник",
        "map_compare": "Карта и сравнение", "map_title": "Посмотрите рекомендации на карте", "map_sub": "Карта использует OpenStreetMap и показывает прототипные профили городов Узбекистана и их текущий рейтинг.",
        "score_compare": "Сравнение оценок", "choose_cities": "Выберите до трёх городов для подробного сравнения", "profile": "Подробный профиль города", "select_city": "Выберите город",
        "filter_cities": "Фильтр городов", "search_placeholder": "Поиск по городу, региону или тегу", "city_explorer": "Исследователь городов", "city_explorer_title": "Просмотрите все прототипные профили",
        "city_explorer_sub": "Каждый профиль использует фотографию, предоставленную для соответствующего города, сохраняя дизайн загруженных карточек.", "region": "Регион", "tags": "Теги", "score": "Прототипная оценка", "dimension": "Параметр",
        "ai_lab": "ИИ-лаборатория", "ai_title": "Как работает рекомендательный модуль", "ai_sub": "Прототип объединяет объяснимые правила, машинное обучение, демонстрационную нейросеть и обучение по обратной связи.",
        "data_archetypes": "Архетипы городов на основе данных", "team": "Команда проекта", "production_warning": "Для промышленного использования замените демонстрационные оценки и синтетические данные на лицензированные, датированные и проверяемые источники. В приложении нет базы данных для счётчика; накопленный счёт хранится внешним сервисом счётчика.",
        "about": "О проекте", "about_title": "Изучать варианты жизни стало проще", "about_sub": "Прототип превращает широкие предпочтения образа жизни в структурированный визуальный процесс сравнения городов.", "included": "Что входит", "ui_preserve": "Сохранение UI-источника", "footer": "ИИ-поиск вариантов для жизни в Узбекистане · портфолио-прототип Streamlit",
        "counter_unavailable": "Счётчик временно недоступен", "like": "Нравится", "not_for_me": "Мне не подходит", "feedback_saved": "Отзыв сохранён для",
        "natural_language": "Понимание естественного языка", "natural_desc": "Обычные фразы вроде «дешёвые продукты» или «красивые закаты» превращаются в измеримые приоритеты.", "map_first": "Исследование с картой", "map_first_desc": "Смотрите города на карте и изучайте их взаимное расположение по Узбекистану.", "transparent": "Прозрачное ранжирование", "transparent_desc": "Параметры предпочтений, текстовое сходство, нейросетевые оценки и обратная связь объединяются в объяснимый балл.", "feedback_learning": "Обучение по отзывам", "feedback_desc": "Лайки и дизлайки обновляют лёгкий bandit-сигнал, помогая прототипу адаптироваться.",
    },
    "uz": {
        "nav_home": "Bosh sahifa", "nav_recommend": "Tavsiyalar", "nav_map": "Xarita va taqqoslash", "nav_cities": "Shaharlar", "nav_ai": "AI laboratoriya", "nav_about": "Loyiha haqida",
        "language": "Til", "app_visits": "Jami tashriflar", "visit_note": "Har bir brauzer sessiyasida bitta tashrif bir marta hisoblanadi.",
        "your_priorities": "Sizning ustuvorliklaringiz", "priority_caption": "Asosiy kiritish matn orqali. Slayderlar alohida ustuvorliklarni kuchaytiradi yoki tuzatadi.",
        "num_recs": "Tavsiyalar soni", "advanced": "Kengaytirilgan ustuvorlik slayderlari", "prototype_note": "Prototip eslatmasi",
        "prototype_copy": "Shahar ballari namuna uchun berilgan, rasmiy statistika emas. Ko‘chishdan oldin uy-joy, ish, xavfsizlik, tibbiyot, viza va huquqiy ma’lumotlarni tekshiring.",
        "featured": "Tanlangan mosliklar", "featured_title": "Siz uchun saralangan turmush variantlari", "featured_sub": "To‘liq tavsiya ish maydonida ishlatiladigan AI dvigateli tanlagan shaharlarning qisqa ko‘rinishi.",
        "experience": "Imkoniyatlar", "experience_title": "Minimal dizayn, foydali intellekt", "experience_sub": "Yuklangan UI ning sokin, tasvirga boy uslubi saqlandi va tajriba shahar izlash atrofida qayta qurildi.",
        "advisor": "AI maslahatchi", "advisor_title": "Ideal joyingizni tasvirlang", "advisor_sub": "Oddiy tilda yozing; batafsil sozlamalar yon panelda va Tavsiyalar bo‘limida mavjud.",
        "lifestyle_request": "Turmush so‘rovi", "find_cities": "Shaharlarimni topish", "updated": "Tavsiyalar yangilandi. Batafsil ma’lumot uchun Tavsiyalar yoki Xarita va taqqoslash bo‘limini oching.",
        "personalized": "Shaxsiy AI maslahatchi", "tell_ai": "AI ga siz uchun nima muhimligini ayting", "tell_ai_sub": "Asosiy signal — matn. Reyting tushunarli turmush ko‘rsatkichlari, NLP o‘xshashligi, neyron model va fikr-mulohazani birlashtiradi.",
        "question": "Qanday joy izlayapsiz?", "find_best": "Eng mos shaharlarni topish", "interpret": "AI so‘rovingizni qanday talqin qildi", "signals": "Aniqlangan signallar:", "preference": "Ustuvorlik", "weight": "Og‘irlik (%)", "top_matches": "Eng mos variantlar",
        "why_fits": "Nega mos keladi", "hybrid": "Gibrid komponentlar", "optional_live": "Qo‘shimcha jonli ma’lumot", "city": "Shahar", "fetch_context": "Ob-havo va shahar xulosasini olish",
        "temp": "Harorat", "feels": "Seziladigan harorat", "conditions": "Holat", "source": "Manba maqolasini ochish",
        "map_compare": "Xarita va taqqoslash", "map_title": "Tavsiyalarni xaritada ko‘ring", "map_sub": "Xarita OpenStreetMap asosida ishlaydi va O‘zbekiston shaharlarining prototip profillari hamda joriy reytinglarini ko‘rsatadi.",
        "score_compare": "Tavsiya ballarini taqqoslash", "choose_cities": "Batafsil taqqoslash uchun uchtagacha shahar tanlang", "profile": "Shahar profili", "select_city": "Shaharni tanlang",
        "filter_cities": "Shaharlarni filtrlash", "search_placeholder": "Shahar, hudud yoki teg bo‘yicha qidirish", "city_explorer": "Shaharlar tadqiqotchisi", "city_explorer_title": "Barcha prototip profillarini ko‘ring",
        "city_explorer_sub": "Har bir profil shu shaharga biriktirilgan suratdan foydalanadi va yuklangan location-card dizaynini saqlaydi.", "region": "Hudud", "tags": "Teglar", "score": "Prototip balli", "dimension": "Yo‘nalish",
        "ai_lab": "AI laboratoriya", "ai_title": "Tavsiya miyasi qanday ishlaydi", "ai_sub": "Prototip tushunarli qoidalar, mashinaviy o‘rganish, namoyish neyron tarmog‘i va fikr-mulohaza orqali o‘rganishni birlashtiradi.",
        "data_archetypes": "Ma’lumotga asoslangan shahar arxetiplari", "team": "Loyiha jamoasi", "production_warning": "Ishlab chiqarish versiyasida namuna ballari va sintetik ma’lumotlarni litsenziyalangan, sanasi ko‘rsatilgan va tekshiriladigan manbalar bilan almashtiring. Ilova tashrif hisoblagichi uchun ma’lumotlar bazasidan foydalanmaydi; jamlanma hisob tashqi counter xizmatida saqlanadi.",
        "about": "Loyiha haqida", "about_title": "Yashash qarorlarini o‘rganish osonroq", "about_sub": "Prototip keng turmush afzalliklarini tuzilgan va vizual shahar taqqoslash jarayoniga aylantiradi.", "included": "Nimalar mavjud", "ui_preserve": "UI manbasini saqlash", "footer": "O‘zbekistonda yashash variantlarini AI yordamida o‘rganish · Streamlit portfolio prototipi",
        "counter_unavailable": "Hisoblagich vaqtincha mavjud emas", "like": "Yoqdi", "not_for_me": "Menga mos emas", "feedback_saved": "Fikr saqlandi:",
        "natural_language": "Tabiiy tilni tushunish", "natural_desc": "«Arzon oziq-ovqat» yoki «ajoyib quyosh botishi» kabi oddiy jumlalarni o‘lchanadigan ustuvorliklarga aylantiradi.", "map_first": "Xarita orqali izlash", "map_first_desc": "Nomzod shaharlarni xaritada ko‘ring va ularning O‘zbekiston bo‘yicha joylashuvini tahlil qiling.", "transparent": "Shaffof baholash", "transparent_desc": "Afzalliklar, matn o‘xshashligi, neyron tarmoq bahosi va fikr-mulohazani tushunarli ballga birlashtiradi.", "feedback_learning": "Fikr-mulohaza orqali o‘rganish", "feedback_desc": "Yoqdi va mos emas signallari yengil bandit ko‘rsatkichini yangilab, prototipni moslashtiradi.",
    },
}

CITY_DESCRIPTIONS = {
    "Samarkand": {"ru": "Крупный город Шёлкового пути, сочетающий всемирно известную архитектуру, активный туризм, красивые вечерние виды, рестораны, университеты и хорошее междугороднее сообщение.", "uz": "Ipak yo‘lining yirik shahri: mashhur me’morchilik, faol turizm, chiroyli kechki manzaralar, restoranlar, universitetlar va yaxshi shaharlararo aloqalarni birlashtiradi."},
    "Bukhara": {"ru": "Компактный исторический город со спокойной атмосферой, сильной культурной идентичностью, красивыми закатами над старой архитектурой и сравнительно доступным прототипным профилем.", "uz": "Sokin muhitga, kuchli madaniy o‘ziga xoslikka, tarixiy me’morchilik ustidagi go‘zal quyosh botishlariga va nisbatan hamyonbop prototip profiliga ega tarixiy shahar."},
    "Khiva": {"ru": "Небольшой город-крепость, особенно сильный по исторической атмосфере, фотографии, видам на закат, пешеходной доступности и спокойному образу жизни.", "uz": "Tarixiy muhit, fotografiya, quyosh botishi manzaralari, piyoda yurish va sokin hayot tarzi bilan ajralib turadigan kichik devorli tarixiy shahar."},
    "Nukus": {"ru": "Более тихая и доступная региональная столица, известная необычным искусством и каракалпакской культурой, пустынным окружением и меньшим числом услуг большого города.", "uz": "O‘ziga xos san’at va Qoraqalpoq madaniyati, cho‘l muhiti hamda yirik shaharlarga qaraganda kamroq xizmatlarga ega, sokinroq va hamyonbop hududiy markaz."},
    "Fergana": {"ru": "Зелёный региональный город Ферганской долины с сильными рынками, доступными продуктами, семейной средой и хорошим доступом к природным местам долины.", "uz": "Bozorlar, hamyonbop oziq-ovqat, oilaviy muhit va vodiydagi tabiiy joylarga yaxshi chiqish imkoniga ega Farg‘ona vodiysidagi yashil shahar."},
    "Andijan": {"ru": "Коммерчески активный город долины с сильными местными рынками, сравнительно доступным прототипным профилем, семейными услугами и региональными деловыми возможностями.", "uz": "Faol mahalliy bozorlar, nisbatan hamyonbop prototip profili, oilaviy xizmatlar va hududiy biznes imkoniyatlariga ega savdo shahri."},
    "Namangan": {"ru": "Большой, но сравнительно спокойный город долины, известный садами, рынками, семейной жизнью и доступом к более зелёным ландшафтам.", "uz": "Bog‘lari, bozorlari, oilaviy hayoti va yashil manzaralarga chiqishi bilan tanilgan, nisbatan sokin vodiy shahri."},
    "Qarshi": {"ru": "Доступный региональный центр с практичным и более спокойным образом жизни, местной промышленностью, железнодорожным сообщением и меньшим числом развлечений.", "uz": "Amaliy va sokinroq hayot tarzi, mahalliy sanoat, temiryo‘l aloqalari hamda ko‘ngilochar imkoniyatlari yirik shaharlarga qaraganda kamroq bo‘lgan hamyonbop hududiy markaz."},
    "Termez": {"ru": "Южный город с важным археологическим наследием, выразительным речным и пустынным светом, тёплой погодой и более медленным региональным ритмом жизни.", "uz": "Muhim arxeologik meros, daryo va cho‘l manzaralarining yorug‘ligi, iliq ob-havo va sokinroq hududiy hayot ritmiga ega janubiy shahar."},
    "Jizzakh": {"ru": "Практичный и доступный город с хорошим доступом к горам и природе, спокойным ритмом и удобным восточно-западным автомобильным и железнодорожным положением.", "uz": "Tog‘ va tabiat maskanlariga qulay chiqish, sokin ritm va sharq-g‘arb avtomobil hamda temiryo‘l yo‘nalishlariga qulay joylashuvga ega amaliy shahar."},
    "Gulistan": {"ru": "Небольшая тихая региональная столица с одним из самых сильных демонстрационных профилей доступности, простым транспортом и ограниченной ночной жизнью.", "uz": "Namuna sifatida eng kuchli hamyonboplik profillaridan biriga, sodda transport aloqalariga va cheklangan tungi hayotga ega kichik, sokin hududiy markaz."},
    "Navoi": {"ru": "Плановый промышленный город с относительно сильным потенциалом занятости, упорядоченной городской средой, парками и полезными воздушными и железнодорожными связями.", "uz": "Nisbatan kuchli bandlik salohiyati, tartibli shahar muhiti, bog‘lar va foydali havo hamda temiryo‘l aloqalariga ega rejalashtirilgan sanoat shahri."},
    "Urgench": {"ru": "Практичная сервисная и транспортная база Хорезма с аэропортом и железной дорогой, местными рынками и удобной близостью к Хиве.", "uz": "Xorazm uchun aeroport va temiryo‘l, mahalliy bozorlar hamda Xivaga yaqinligi bilan qulay xizmat va transport markazi."},
    "Kokand": {"ru": "Исторический город Ферганской долины с дворцовой архитектурой, традиционными ремёслами, рынками и сбалансированным сочетанием доступности и культурной жизни.", "uz": "Saroy me’morchiligi, an’anaviy hunarmandchilik, bozorlar va hamyonboplik bilan madaniy qiziqishning muvozanatli uyg‘unligiga ega tarixiy vodiy shahri."},
    "Shahrisabz": {"ru": "Небольшой исторический город к югу от Самарканда с горными пейзажами, крупным тимуридским наследием, красивыми закатами и спокойным образом жизни.", "uz": "Samarqand janubidagi tog‘ manzaralari, Temuriylar merosi, chiroyli quyosh botishlari va sokin hayot tarziga ega kichik tarixiy shahar."},
    "Tashkent": {"ru": "Крупнейший городской центр Узбекистана с сильнейшим сочетанием рабочих мест, университетов, медицины, общественного транспорта, ресторанов и развлечений, но с более высоким прототипным профилем стоимости и более быстрым ритмом.", "uz": "O‘zbekistonning eng yirik shahri: ish o‘rinlari, universitetlar, tibbiyot, jamoat transporti, restoranlar va ko‘ngilochar imkoniyatlari kuchli, biroq prototip xarajat profili yuqoriroq va hayot ritmi tezroq."},
}

# -----------------------------------------------------------------------------
# Persistent visit counter (no database in the repository)
# CounterAPI is a public external counter service.  We increment once per
# Streamlit browser session, then display the cumulative value throughout the app.
# -----------------------------------------------------------------------------
COUNTER_NAMESPACE = "uzbekistan-ai-map-living-advisor.streamlit.app"
COUNTER_ACTION = "view"
COUNTER_KEY = "app-users"
COUNTER_URL = f"https://counterapi.com/api/{COUNTER_NAMESPACE}/{COUNTER_ACTION}/{COUNTER_KEY}"


def t(key: str, lang: str | None = None) -> str:
    lang = lang or st.session_state.get("language", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))


def city_description(row: pd.Series, lang: str | None = None) -> str:
    lang = lang or st.session_state.get("language", "en")
    city = str(row.get("city", ""))
    if lang in {"ru", "uz"} and city in CITY_DESCRIPTIONS:
        return CITY_DESCRIPTIONS[city][lang]
    return str(row.get("description", ""))


def localized_feature_label(feature: str, lang: str | None = None) -> str:
    lang = lang or st.session_state.get("language", "en")
    if lang == "en":
        return FEATURE_LABELS[feature]
    maps = {
        "ru": {"affordability":"Доступность проживания","grocery_affordability":"Доступность продуктов","entertainment":"Развлечения","sunset_scenery":"Закаты и пейзажи","safety":"Безопасность","jobs":"Работа и карьера","internet":"Интернет и удалённая работа","healthcare":"Медицина","education":"Образование","heritage":"История и культура","nature":"Природа","climate_comfort":"Комфорт климата","quietness":"Спокойный образ жизни","mobility":"Транспорт"},
        "uz": {"affordability":"Yashash narxi","grocery_affordability":"Oziq-ovqat narxi","entertainment":"Ko‘ngilochar imkoniyatlar","sunset_scenery":"Quyosh botishi va manzara","safety":"Xavfsizlik","jobs":"Ish va karyera","internet":"Internet va masofaviy ish","healthcare":"Tibbiyot","education":"Ta’lim","heritage":"Tarix va madaniyat","nature":"Tabiat","climate_comfort":"Iqlim qulayligi","quietness":"Sokin hayot","mobility":"Transport"},
    }
    return maps.get(lang, {}).get(feature, FEATURE_LABELS[feature])


def record_app_visit() -> int | None:
    """Increment the persistent counter once per Streamlit session."""
    if st.session_state.get("usage_counted"):
        return st.session_state.get("usage_count")
    try:
        response = requests.get(
            COUNTER_URL,
            params={"startNumber": 0},
            timeout=5,
            headers={"User-Agent": "Akbarxon-AI-Living-Advisor/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        value = int(payload.get("value"))
        if value < 1:
            return None
        st.session_state["usage_counted"] = True
        st.session_state["usage_count"] = value
        return value
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return None


def render_usage_counter(count: int | None) -> None:
    display = f"{count:,}" if isinstance(count, int) and count > 0 else "—"
    label = t("app_visits")
    note = t("visit_note")
    st.markdown(
        f"<div class='usage-counter'><span class='usage-icon'>👥</span><div><div class='usage-label'>{html.escape(label)}</div><div class='usage-value'>{display}</div><div class='usage-note'>{html.escape(note)}</div></div></div>",
        unsafe_allow_html=True,
    )


def city_image_file(city: object) -> str:
    """Return the repository-relative image assigned to a city."""
    filename = CITY_IMAGE_FILES.get(str(city).strip(), FALLBACK_CITY_IMAGE)
    return filename if (ASSET_DIR / filename).exists() else FALLBACK_CITY_IMAGE


def city_image_path(city: object) -> Path:
    return ASSET_DIR / city_image_file(city)


def image_data_uri(filename: str) -> str:
    path = ASSET_DIR / filename
    if not path.exists():
        return ""
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@st.cache_data
def load_city_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(50).clip(0, 100)
    return df


def city_document(row: pd.Series) -> str:
    feature_words = " ".join(
        FEATURE_LABELS[column] for column in FEATURE_COLUMNS if float(row[column]) >= 72
    )
    return " ".join(
        [
            str(row.get("city", "")),
            str(row.get("region", "")),
            str(row.get("tags", "")),
            str(row.get("description", "")),
            feature_words,
        ]
    )


@st.cache_resource
def build_text_index(documents: Tuple[str, ...]) -> Tuple[TfidfVectorizer, object]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def parse_preference_weights(query: str) -> Tuple[Dict[str, float], List[str]]:
    text = re.sub(r"\s+", " ", query.lower()).strip()
    weights = {feature: 0.0 for feature in FEATURE_COLUMNS}
    matches: List[str] = []

    for feature, phrases in SYNONYMS.items():
        for phrase in phrases:
            if phrase in text:
                weights[feature] += 1.4 if " " in phrase else 1.0
                matches.append(f"{localized_feature_label(feature)} <- '{phrase}'")

    if "family" in text or "children" in text or "kids" in text:
        weights["safety"] += 1.1
        weights["education"] += 0.9
        weights["healthcare"] += 0.6
        matches.append("Family intent -> safety, education, healthcare")
    if "retire" in text or "retirement" in text:
        weights["affordability"] += 0.8
        weights["quietness"] += 1.0
        weights["healthcare"] += 0.8
        matches.append("Retirement intent -> affordability, quietness, healthcare")
    if "young professional" in text or "professional" in text:
        weights["jobs"] += 1.0
        weights["internet"] += 0.7
        weights["entertainment"] += 0.5
        matches.append("Professional intent -> jobs, internet, entertainment")
    if "tourist" in text or "tourism" in text or "visit" in text:
        weights["heritage"] += 0.9
        weights["entertainment"] += 0.5
        weights["mobility"] += 0.4
        matches.append("Tourism intent -> heritage, activities, transportation")

    total = sum(weights.values())
    if total <= 0:
        defaults = {
            "affordability": 1.0,
            "safety": 1.0,
            "healthcare": 0.8,
            "internet": 0.7,
            "mobility": 0.6,
            "entertainment": 0.5,
        }
        weights.update(defaults)
        matches.append("No strong keyword found -> balanced default profile")
        total = sum(weights.values())

    return {key: value / total for key, value in weights.items()}, matches


def combine_weights(text_weights: Dict[str, float], slider_weights: Dict[str, float]) -> Dict[str, float]:
    combined = {
        feature: text_weights.get(feature, 0.0) + slider_weights.get(feature, 0.0) / 5.0
        for feature in FEATURE_COLUMNS
    }
    total = sum(combined.values()) or 1.0
    return {feature: value / total for feature, value in combined.items()}


def load_feedback_state(cities: Iterable[str]) -> Dict[str, Dict[str, int]]:
    default = {str(city): {"likes": 0, "dislikes": 0} for city in cities}
    if not FEEDBACK_PATH.exists():
        return default
    try:
        stored = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
        for city in default:
            values = stored.get(city, {})
            default[city]["likes"] = int(values.get("likes", 0))
            default[city]["dislikes"] = int(values.get("dislikes", 0))
    except (OSError, ValueError, TypeError):
        pass
    return default


def save_feedback_state(state: Dict[str, Dict[str, int]]) -> None:
    try:
        FEEDBACK_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


def bandit_posterior(city: str, state: Dict[str, Dict[str, int]]) -> float:
    values = state.get(city, {"likes": 0, "dislikes": 0})
    likes = max(0, int(values.get("likes", 0)))
    dislikes = max(0, int(values.get("dislikes", 0)))
    return (likes + 1.0) / (likes + dislikes + 2.0)


def make_dnn_training_data(df: pd.DataFrame, n_samples: int = 1400) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    features = df[FEATURE_COLUMNS].to_numpy(dtype=float) / 100.0
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    for _ in range(n_samples):
        city_vector = features[rng.integers(0, len(features))]
        preference = rng.dirichlet(np.ones(len(FEATURE_COLUMNS)) * 0.75)
        interaction = preference * city_vector
        utility = float(np.dot(preference, city_vector))

        idx = {name: FEATURE_COLUMNS.index(name) for name in FEATURE_COLUMNS}
        utility += 0.09 * min(
            preference[idx["heritage"]] * city_vector[idx["heritage"]],
            preference[idx["entertainment"]] * city_vector[idx["entertainment"]],
        )
        utility += 0.08 * min(
            preference[idx["affordability"]] * city_vector[idx["affordability"]],
            preference[idx["grocery_affordability"]] * city_vector[idx["grocery_affordability"]],
        )
        utility += 0.07 * min(
            preference[idx["jobs"]] * city_vector[idx["jobs"]],
            preference[idx["internet"]] * city_vector[idx["internet"]],
        )
        utility += 0.06 * min(
            preference[idx["nature"]] * city_vector[idx["nature"]],
            preference[idx["sunset_scenery"]] * city_vector[idx["sunset_scenery"]],
        )
        utility += rng.normal(0, 0.025)

        x_rows.append(np.concatenate([preference, city_vector, interaction]))
        like_probability = 1.0 / (1.0 + math.exp(-(utility - 0.73) / 0.055))
        y_rows.append(int(rng.random() < like_probability))

    return np.vstack(x_rows), np.asarray(y_rows, dtype=int)


@st.cache_resource
def train_demo_dnn(data_signature: str, _df: pd.DataFrame) -> Pipeline:
    del data_signature
    x_train, y_train = make_dnn_training_data(_df)
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "dnn",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    alpha=0.001,
                    learning_rate_init=0.002,
                    max_iter=220,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def dnn_like_probabilities(model: Pipeline, df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    preference = np.array([weights[feature] for feature in FEATURE_COLUMNS], dtype=float)
    city_features = df[FEATURE_COLUMNS].to_numpy(dtype=float) / 100.0
    x = np.hstack(
        [
            np.repeat(preference.reshape(1, -1), len(df), axis=0),
            city_features,
            city_features * preference,
        ]
    )
    return model.predict_proba(x)[:, 1]


def rank_cities(
    df: pd.DataFrame,
    query: str,
    slider_weights: Dict[str, float],
    feedback_state: Dict[str, Dict[str, int]],
    top_n: int,
) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    text_weights, matches = parse_preference_weights(query)
    weights = combine_weights(text_weights, slider_weights)

    documents = tuple(city_document(row) for _, row in df.iterrows())
    vectorizer, city_matrix = build_text_index(documents)
    query_vector = vectorizer.transform([query or "balanced affordable safe city"])
    text_similarity = cosine_similarity(query_vector, city_matrix).ravel()

    feature_matrix = df[FEATURE_COLUMNS].to_numpy(dtype=float) / 100.0
    weight_vector = np.array([weights[column] for column in FEATURE_COLUMNS], dtype=float)
    preference_score = feature_matrix @ weight_vector

    signature = str(hash(tuple(np.round(feature_matrix.ravel(), 4))))
    dnn_model = train_demo_dnn(signature, df)
    dnn_score = dnn_like_probabilities(dnn_model, df, weights)
    posterior = np.array([bandit_posterior(city, feedback_state) for city in df["city"]], dtype=float)

    final_score = 0.68 * preference_score + 0.14 * text_similarity + 0.13 * dnn_score + 0.05 * posterior

    ranked = df.copy()
    ranked["preference_score"] = preference_score * 100
    ranked["text_similarity"] = text_similarity * 100
    ranked["dnn_like_probability"] = dnn_score * 100
    ranked["feedback_posterior"] = posterior * 100
    ranked["match_score"] = final_score * 100
    ranked = ranked.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)
    return ranked, weights, matches


def top_reasons(row: pd.Series, weights: Dict[str, float], n: int = 4) -> List[str]:
    contributions = []
    for feature in FEATURE_COLUMNS:
        contributions.append((weights.get(feature, 0.0) * float(row[feature]), feature, float(row[feature])))
    contributions.sort(reverse=True)
    return [f"{localized_feature_label(feature)}: {value:.0f}/100" for _, feature, value in contributions[:n]]


def cluster_cities(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    model = KMeans(n_clusters=4, random_state=42, n_init=20)
    output["cluster_id"] = model.fit_predict(df[FEATURE_COLUMNS])
    summaries = output.groupby("cluster_id")[FEATURE_COLUMNS].mean()
    labels: Dict[int, str] = {}
    for cluster_id, values in summaries.iterrows():
        if values["jobs"] + values["healthcare"] + values["education"] >= 230:
            label = "Career and services hub"
        elif values["heritage"] + values["entertainment"] >= 145:
            label = "Culture and tourism hub"
        elif values["affordability"] + values["quietness"] >= 155:
            label = "Affordable quiet city"
        else:
            label = "Balanced regional center"
        labels[int(cluster_id)] = label
    output["city_archetype"] = output["cluster_id"].map(labels)
    return output


def build_map(df: pd.DataFrame, ranked: pd.DataFrame | None = None) -> folium.Map:
    map_object = folium.Map(location=[41.2, 64.6], zoom_start=5, tiles="OpenStreetMap", control_scale=True)
    ranked_lookup = {}
    if ranked is not None:
        ranked_lookup = {
            city: (index + 1, float(score))
            for index, (city, score) in enumerate(zip(ranked["city"], ranked["match_score"]))
        }

    for _, row in df.iterrows():
        city = str(row["city"])
        rank_text = ""
        icon_color = "darkgreen"
        if city in ranked_lookup:
            rank, score = ranked_lookup[city]
            rank_text = f"<br><b>Recommendation rank:</b> #{rank}<br><b>Match:</b> {score:.1f}/100"
            icon_color = "green" if rank == 1 else "cadetblue"
        popup = folium.Popup(
            f"<b>{city}</b><br>{row['region']}<br>{row['description']}{rank_text}",
            max_width=330,
        )
        folium.Marker(
            location=[float(row["lat"]), float(row["lon"])],
            tooltip=city,
            popup=popup,
            icon=folium.Icon(color=icon_color, icon="home"),
        ).add_to(map_object)
    return map_object


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_weather(lat: float, lon: float) -> Dict[str, object]:
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,sunset",
            "forecast_days": 3,
            "timezone": "auto",
        },
        timeout=7,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_wikipedia_summary(city: str) -> Dict[str, object]:
    response = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(city)}",
        timeout=7,
        headers={"User-Agent": "AkbarxonLivingAdvisorPrototype/2.0"},
    )
    response.raise_for_status()
    return response.json()


def weather_code_label(code: int | float | None) -> str:
    if code is None:
        return "Unknown"
    code = int(code)
    if code == 0:
        return "Clear sky"
    if code in {1, 2, 3}:
        return "Partly cloudy"
    if code in {45, 48}:
        return "Fog"
    if 51 <= code <= 67:
        return "Rain or drizzle"
    if 71 <= code <= 77:
        return "Snow"
    if 80 <= code <= 82:
        return "Rain showers"
    if code >= 95:
        return "Thunderstorm"
    return "Mixed conditions"


def radar_figure(row: pd.Series, features: List[str]) -> go.Figure:
    values = [float(row[feature]) for feature in features]
    labels = [localized_feature_label(feature) for feature in features]
    fig = go.Figure(
        data=[go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], fill="toself", name=str(row["city"]))]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=28, r=28, t=28, b=28),
        height=420,
    )
    return fig


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #fcfcfc;
          --fg: #333333;
          --muted: #7c7c7c;
          --primary: #8ea89b;
          --primary-dark: #536f61;
          --accent: #eef5f1;
          --border: #e9e9e9;
          --card: #ffffff;
        }
        html {scroll-behavior: smooth;}
        .stApp {background: var(--bg); color: var(--fg);}

        /*
         * Streamlit Community Cloud keeps its native toolbar fixed at the top.
         * The previous 1rem top padding pulled the custom brand/navigation under
         * that toolbar. Reserve a safe top zone for the main canvas instead.
         */
        .block-container {
          max-width: 1280px;
          padding-top: 4.75rem !important;
          padding-bottom: 4rem;
        }
        [data-testid="stMainBlockContainer"] {
          max-width: 1280px;
          padding-top: 4.75rem !important;
          padding-bottom: 4rem;
        }
        h1, h2, h3, h4 {letter-spacing: -0.02em;}
        [data-testid="stSidebar"] {background: #f9faf9; border-right: 1px solid var(--border);}
        [data-testid="stSidebar"] .block-container {padding-top: 1rem !important;}
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-top: .35rem;}

        .top-brand {
          display:flex;
          align-items:center;
          justify-content:space-between;
          gap:1rem;
          min-height:42px;
          padding:.35rem .15rem 1rem .15rem;
          position:relative;
          z-index:1;
        }
        .brand-left {display:flex; align-items:center; gap:.7rem;}
        .brand-mark {width:34px; height:34px; border-radius:50%; display:grid; place-items:center; border:1px solid var(--border); background:white; font-size:18px;}
        .brand-name {font-size:1.02rem; font-weight:500; letter-spacing:.01em;}
        .team-mini {font-size:.78rem; color:var(--muted); text-align:right; line-height:1.45;}

        .hero-shell {position:relative; min-height:565px; border-radius:18px; overflow:hidden; margin: .4rem 0 4rem 0; box-shadow: 0 12px 38px rgba(0,0,0,.08);}
        .hero-shell img {width:100%; height:565px; object-fit:cover; display:block;}
        .hero-overlay {position:absolute; inset:0; background:linear-gradient(180deg, rgba(0,0,0,.05) 0%, rgba(0,0,0,.42) 100%);}
        .hero-copy {position:absolute; left:42px; bottom:54px; color:white; max-width:560px;}
        .hero-eyebrow {font-size:.74rem; text-transform:uppercase; letter-spacing:.18em; margin-bottom:1rem; opacity:.92;}
        .hero-title {font-size:clamp(2.45rem,5vw,4.7rem); font-weight:300; line-height:.96; margin:0 0 1.2rem 0;}
        .hero-sub {font-size:1rem; line-height:1.65; max-width:520px; opacity:.94; margin-bottom:1.4rem;}
        .hero-pill {display:inline-block; background:#fff; color:#303030; padding:.8rem 1.15rem; border-radius:999px; font-size:.82rem; text-decoration:none;}
        .hero-bars {position:absolute; bottom:22px; left:42px; right:42px; display:flex; gap:8px;}
        .hero-bars span {height:2px; flex:1; background:rgba(255,255,255,.35);}
        .hero-bars span:first-child {background:white;}

        .section-head {text-align:center; margin: 1.5rem 0 2.5rem 0;}
        .eyebrow {font-size:.7rem; text-transform:uppercase; letter-spacing:.16em; color:var(--muted); margin-bottom:.7rem;}
        .section-title {font-size:2rem; font-weight:300; margin:0 0 .7rem 0;}
        .section-sub {font-size:.93rem; color:var(--muted); max-width:620px; margin:0 auto; line-height:1.6;}

        .city-card-html {overflow:hidden; border:1px solid var(--border); background:var(--card); border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,.055); min-height:100%; height:100%; margin-bottom:1.1rem;}
        .city-card-html img {width:100%; height:205px; object-fit:cover; display:block;}
        .city-card-body {padding:1.25rem 1.25rem 1.35rem 1.25rem;}
        .city-card-top {display:flex; justify-content:space-between; gap:1rem; align-items:flex-start;}
        .city-card-title {font-size:1.05rem; font-weight:500; margin:0;}
        .city-card-score {font-size:.76rem; padding:.3rem .55rem; border-radius:999px; background:var(--accent); color:var(--primary-dark); white-space:nowrap;}
        .city-region {font-size:.78rem; color:var(--muted); margin:.3rem 0 .85rem 0;}
        .city-desc {font-size:.86rem; color:#5e5e5e; line-height:1.55; min-height:5.35em; overflow:hidden; display:-webkit-box; -webkit-line-clamp:4; -webkit-box-orient:vertical;}
        .chips {display:flex; flex-wrap:wrap; gap:6px; margin-top:.8rem;}
        .chip {font-size:.64rem; text-transform:uppercase; letter-spacing:.08em; background:var(--accent); padding:.3rem .45rem; border-radius:4px; color:#4f655a;}

        .experience-row {display:flex; align-items:center; gap:1rem; padding:1.2rem 1.25rem; border:1px solid rgba(255,255,255,.8); background:rgba(255,255,255,.72); border-radius:10px; margin:.75rem 0; box-shadow:0 2px 12px rgba(0,0,0,.035); transition:.25s ease;}
        .experience-row:hover {transform:translateY(-3px); box-shadow:0 10px 25px rgba(0,0,0,.07);}
        .experience-icon {width:46px; height:46px; border-radius:50%; display:grid; place-items:center; background:var(--accent); font-size:21px; flex:0 0 auto;}
        .experience-title {font-size:.92rem; font-weight:500; margin-bottom:.25rem;}
        .experience-desc {font-size:.82rem; color:var(--muted); line-height:1.5;}

        .advisor-card {border:1px solid var(--border); border-radius:14px; background:white; padding:1.5rem; box-shadow:0 8px 26px rgba(0,0,0,.045);}
        .result-card {border:1px solid var(--border); border-radius:12px; background:white; padding:1.25rem; margin:.75rem 0;}
        .result-rank {font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; color:var(--muted);}
        .result-title {font-size:1.3rem; font-weight:400; margin:.3rem 0;}
        .score-badge {display:inline-block; background:var(--accent); color:var(--primary-dark); border-radius:999px; padding:.35rem .65rem; font-size:.78rem; font-weight:600;}
        .result-note {font-size:.78rem; color:var(--muted); line-height:1.55;}

        .team-card {padding:1.05rem 1rem; border:1px solid var(--border); border-radius:12px; background:white; margin-bottom:1rem;}

        .usage-counter {display:flex; align-items:center; gap:.7rem; padding:.75rem .9rem; margin:.35rem 0 1rem 0; border:1px solid #dfe8e3; border-radius:12px; background:linear-gradient(135deg,#f3f8f5,#ffffff);}
        .usage-icon {width:30px; height:30px; border-radius:50%; display:grid; place-items:center; background:#e7f0eb; font-size:16px; flex:0 0 auto;}
        .usage-label {font-size:.67rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted);}
        .usage-value {font-size:1.12rem; font-weight:600; line-height:1.15; color:var(--primary-dark);}
        .usage-note {font-size:.64rem; color:var(--muted); margin-top:.1rem;}
        .team-title {font-size:.85rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin-bottom:.65rem;}
        .team-name {font-size:.9rem; line-height:1.65;}
        .prototype-note {font-size:.8rem; color:#666; line-height:1.55; padding:.85rem; background:#f1f6f3; border-radius:10px; border:1px solid #e3eee8;}

        .ai-card {padding:1.3rem; border:1px solid var(--border); border-radius:12px; background:white; min-height:180px;}
        .ai-num {font-size:.7rem; letter-spacing:.14em; color:var(--muted); text-transform:uppercase;}
        .ai-title {font-size:1rem; font-weight:500; margin:.45rem 0;}
        .ai-desc {font-size:.84rem; line-height:1.58; color:#6b6b6b;}

        .footer-shell {background:#2f302f; color:#f6f6f6; border-radius:14px; padding:2rem 2.2rem; margin-top:4rem;}
        .footer-brand {font-size:1rem; font-weight:500; margin-bottom:.5rem;}
        .footer-copy {font-size:.8rem; color:rgba(255,255,255,.68); line-height:1.6;}
        .footer-rule {border-top:1px solid rgba(255,255,255,.16); margin:1.4rem 0 1rem 0;}

        div[data-testid="stRadio"] > div {gap:.15rem;}
        div[data-testid="stRadio"] label {border-radius:999px; padding:.18rem .55rem;}
        div[data-testid="stButton"] button {border-radius:999px; font-weight:500;}
        div[data-testid="stFormSubmitButton"] button {border-radius:999px;}
        .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {border-radius:10px;}

        @media (max-width: 800px) {
          .hero-shell, .hero-shell img {min-height:470px; height:470px;}
          .hero-copy {left:24px; right:24px; bottom:46px;}
          .hero-bars {left:24px; right:24px;}
          .team-mini {display:none;}
          .hero-title {font-size:2.7rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_brand() -> None:
    st.markdown(
        """
        <div class="top-brand">
          <div class="brand-left">
            <div class="brand-mark">🧭</div>
            <div class="brand-name">Akbarxon AI Living Advisor</div>
          </div>
          <div class="team-mini">Author: Akbarxon Nasirov<br>Mentor: Dr. Qingyang Xiao</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    hero = image_data_uri("hero-camping.jpg")
    lang = st.session_state.get("language", "en")
    hero_content = {
        "en": ("AI-powered city discovery · Uzbekistan", "Find a place<br>that fits your life", "Describe the lifestyle you want. The advisor translates your words into priorities, ranks Uzbekistan cities, maps the results, and learns from feedback.", "Start Exploring"),
        "ru": ("ИИ-поиск городов · Узбекистан", "Найдите место,<br>которое подходит вам", "Опишите желаемый образ жизни. Консультант преобразует ваши слова в приоритеты, ранжирует города Узбекистана, показывает результаты на карте и обучается на отзывах.", "Начать исследование"),
        "uz": ("AI yordamida shahar izlash · O‘zbekiston", "Hayotingizga mos<br>joyni toping", "Istagan turmush tarzingizni tasvirlang. Maslahatchi so‘zlaringizni ustuvorliklarga aylantiradi, O‘zbekiston shaharlarini saralaydi, natijalarni xaritada ko‘rsatadi va fikrlardan o‘rganadi.", "Izlashni boshlash"),
    }
    hero_eyebrow, hero_title, hero_subtitle, hero_button = hero_content.get(lang, hero_content["en"])
    st.markdown(
        f"""
        <div class="hero-shell">
          <img src="{hero}" alt="Lifestyle landscape design visual">
          <div class="hero-overlay"></div>
          <div class="hero-copy">
            <div class="hero-eyebrow">{html.escape(hero_eyebrow)}</div>
            <div class="hero-title">{hero_title}</div>
            <div class="hero-sub">{html.escape(hero_subtitle)}</div>
            <a class="hero-pill" href="#advisor">{html.escape(hero_button)} &nbsp;→</a>
          </div>
          <div class="hero-bars"><span></span><span></span><span></span><span></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_head(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-head">
          <div class="eyebrow">{eyebrow}</div>
          <div class="section-title">{title}</div>
          <div class="section-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_city_card_html(row: pd.Series, rank: int | None = None) -> str:
    """Build one self-contained card without Markdown blank-line parsing issues."""
    city = html.escape(str(row.get("city", "Unknown city")))
    region = html.escape(str(row.get("region", "")))
    description = html.escape(city_description(row))
    image = html.escape(image_data_uri(city_image_file(row.get("city", ""))), quote=True)
    tags = [item.strip() for item in str(row.get("tags", "")).split(",") if item.strip()][:3]
    chips = "".join(f'<span class="chip">{html.escape(tag)}</span>' for tag in tags)

    raw_score = row.get("match_score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    score_html = (
        f'<span class="city-card-score">{score:.1f}</span>'
        if math.isfinite(score) and score > 0
        else ""
    )
    rank_text = f"#{rank} · " if rank else ""

    # Join the HTML fragments directly. A blank line inside a raw Markdown HTML
    # block can terminate the block and expose closing tags as visible text.
    return "".join(
        [
            '<div class="city-card-html">',
            f'<img src="{image}" alt="{city} city photo" loading="lazy">',
            '<div class="city-card-body">',
            '<div class="city-card-top">',
            f'<div class="city-card-title">{rank_text}{city}</div>',
            score_html,
            '</div>',
            f'<div class="city-region">&#128205; {region}</div>',
            f'<div class="city-desc">{description}</div>',
            f'<div class="chips">{chips}</div>',
            '</div>',
            '</div>',
        ]
    )


def render_city_card(row: pd.Series, rank: int | None = None) -> None:
    # st.html renders the card as HTML rather than asking the Markdown parser to
    # infer where the raw HTML block begins and ends.
    st.html(build_city_card_html(row, rank=rank))


def render_experience_rows() -> None:
    items = [
        ("🧠", "Natural-language understanding", "Turn everyday phrases such as 'cheap groceries' or 'great sunsets' into measurable lifestyle priorities."),
        ("🗺️", "Map-first exploration", "See candidate cities geographically and inspect how recommendations relate across Uzbekistan."),
        ("📊", "Transparent scoring", "Combine preference features, text similarity, neural-network estimates, and feedback into an explainable score."),
        ("👍", "Feedback learning", "Likes and dislikes update a lightweight bandit signal so the prototype can adapt over time."),
    ]
    for icon, title, desc in items:
        st.markdown(
            f"""
            <div class="experience-row">
              <div class="experience-icon">{icon}</div>
              <div><div class="experience-title">{title}</div><div class="experience-desc">{desc}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_feedback_buttons(city: str, feedback_state: Dict[str, Dict[str, int]], key_prefix: str) -> None:
    left, right, stats = st.columns([1, 1, 2.4])
    if left.button(f"👍 {t('like')}", key=f"{key_prefix}_like_{city}", use_container_width=True):
        feedback_state[city]["likes"] += 1
        save_feedback_state(feedback_state)
        st.toast(f"{t('feedback_saved')} {city}")
        st.rerun()
    if right.button(f"👎 {t('not_for_me')}", key=f"{key_prefix}_dislike_{city}", use_container_width=True):
        feedback_state[city]["dislikes"] += 1
        save_feedback_state(feedback_state)
        st.toast(f"{t('feedback_saved')} {city}")
        st.rerun()
    values = feedback_state[city]
    stats.caption(
        f"Feedback: {values['likes']} likes · {values['dislikes']} dislikes · "
        f"bandit confidence {bandit_posterior(city, feedback_state):.2f}"
    )


def ensure_default_results(
    df: pd.DataFrame,
    slider_weights: Dict[str, float],
    feedback_state: Dict[str, Dict[str, int]],
    top_n: int,
) -> None:
    if "ranked_results" not in st.session_state:
        query = "I want an affordable city with cheap groceries, beautiful sunsets, and good entertainment."
        ranked, weights, matches = rank_cities(df, query, slider_weights, feedback_state, top_n)
        st.session_state["ranked_results"] = ranked
        st.session_state["active_weights"] = weights
        st.session_state["query_matches"] = matches
        st.session_state["active_query"] = query


def render_home(
    df: pd.DataFrame,
    slider_weights: Dict[str, float],
    feedback_state: Dict[str, Dict[str, int]],
    top_n: int,
) -> None:
    ensure_default_results(df, slider_weights, feedback_state, top_n)
    render_hero()

    render_section_head(
        t("featured"),
        t("featured_title"),
        t("featured_sub"),
    )
    ranked = st.session_state["ranked_results"].head(3)
    cols = st.columns(3)
    for idx, (col, (_, row)) in enumerate(zip(cols, ranked.iterrows())):
        with col:
            render_city_card(row, rank=idx + 1)

    st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
    render_section_head(
        t("experience"),
        t("experience_title"),
        t("experience_sub"),
    )
    render_experience_rows()

    st.markdown('<div id="advisor"></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    render_section_head(
        t("advisor"),
        t("advisor_title"),
        t("advisor_sub"),
    )
    with st.form("home_quick_advisor"):
        quick_query = st.text_area(
            "Lifestyle request",
            value=st.session_state.get("active_query", "I want an affordable city with good food prices and beautiful scenery."),
            height=115,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Find my cities", type="primary", use_container_width=True)
    if submitted:
        ranked, weights, matches = rank_cities(df, quick_query, slider_weights, feedback_state, top_n)
        st.session_state["ranked_results"] = ranked
        st.session_state["active_weights"] = weights
        st.session_state["query_matches"] = matches
        st.session_state["active_query"] = quick_query
        st.success("Recommendations updated. Open the Recommend or Map & Compare workspace for full details.")
        st.dataframe(ranked[["city", "region", "match_score"]].head(5), hide_index=True, use_container_width=True)


def render_recommend(
    df: pd.DataFrame,
    slider_weights: Dict[str, float],
    feedback_state: Dict[str, Dict[str, int]],
    top_n: int,
) -> None:
    render_section_head(
        t("personalized"),
        t("tell_ai"),
        t("tell_ai_sub"),
    )

    query = st.text_area(
        t("question"),
        value=st.session_state.get(
            "active_query",
            "I want an affordable city with cheap groceries, beautiful sunsets, and good entertainment.",
        ),
        height=120,
        help="Examples: family-friendly and safe; best for remote work; historic and walkable; quiet retirement city.",
    )
    if st.button(t("find_best"), type="primary", use_container_width=True):
        ranked, weights, matches = rank_cities(df, query, slider_weights, feedback_state, top_n)
        st.session_state["ranked_results"] = ranked
        st.session_state["active_weights"] = weights
        st.session_state["query_matches"] = matches
        st.session_state["active_query"] = query

    ensure_default_results(df, slider_weights, feedback_state, top_n)
    ranked = st.session_state["ranked_results"]
    weights = st.session_state["active_weights"]
    matches = st.session_state["query_matches"]

    with st.expander(t("interpret"), expanded=False):
        st.write(t("signals"))
        for match in matches:
            st.write(f"- {match}")
        weight_table = pd.DataFrame(
            {
                t("preference"): [localized_feature_label(key) for key in FEATURE_COLUMNS],
                "Weight (%)": [round(weights[key] * 100, 1) for key in FEATURE_COLUMNS],
            }
        ).sort_values("Weight (%)", ascending=False)
        st.dataframe(weight_table, hide_index=True, use_container_width=True)

    st.markdown(f"### {t('top_matches')}")
    for index, row in ranked.iterrows():
        reasons = top_reasons(row, weights)
        image_path = city_image_path(row["city"])
        left, right = st.columns([1, 2.2])
        with left:
            st.image(image_path, use_container_width=True, caption=f"{row['city']} city photo")
        with right:
            st.markdown(
                f"""
                <div class="result-card">
                  <div class="result-rank">Recommendation #{index + 1}</div>
                  <div class="result-title">{row['city']} &nbsp; <span class="score-badge">{row['match_score']:.1f}/100 match</span></div>
                  <div class="city-region">{row['region']}</div>
                  <p>{row['description']}</p>
                  <p><b>Why it fits:</b> {' · '.join(reasons)}</p>
                  <div class="result-note">Hybrid components: preference {row['preference_score']:.1f}, text {row['text_similarity']:.1f}, neural model {row['dnn_like_probability']:.1f}, feedback {row['feedback_posterior']:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_feedback_buttons(str(row["city"]), feedback_state, f"rec_{index}")

    st.markdown(f"### {t('optional_live')}")
    selected_live_city = st.selectbox(t("city"), ranked["city"].tolist(), key="live_city")
    if st.button(t("fetch_context")):
        city_row = df.loc[df["city"] == selected_live_city].iloc[0]
        with st.spinner("Retrieving public context..."):
            try:
                wiki = fetch_wikipedia_summary(selected_live_city)
                weather = fetch_live_weather(float(city_row["lat"]), float(city_row["lon"]))
                current = weather.get("current", {})
                c1, c2, c3 = st.columns(3)
                c1.metric(t("temp"), f"{current.get('temperature_2m', '—')} °C")
                c2.metric(t("feels"), f"{current.get('apparent_temperature', '—')} °C")
                c3.metric(t("conditions"), weather_code_label(current.get("weather_code")))
                st.write(wiki.get("extract", "No summary was returned."))
                source_url = wiki.get("content_urls", {}).get("desktop", {}).get("page")
                if source_url:
                    st.link_button(t("source"), source_url)
            except requests.RequestException as exc:
                st.warning(f"Live data is temporarily unavailable: {exc}")


def render_map_compare(df: pd.DataFrame, slider_weights, feedback_state, top_n) -> None:
    ensure_default_results(df, slider_weights, feedback_state, top_n)
    ranked = st.session_state["ranked_results"]
    render_section_head(
        t("map_compare"),
        t("map_title"),
        t("map_sub"),
    )
    st_folium(build_map(df, ranked), width=None, height=590, returned_objects=[])

    st.markdown(f"### {t('score_compare')}")
    chart_df = ranked[["city", "match_score"]].sort_values("match_score")
    fig = px.bar(chart_df, x="match_score", y="city", orientation="h", range_x=[0, 100])
    fig.update_layout(height=390, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Match score", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    compare_names = st.multiselect(
        t("choose_cities"),
        df["city"].tolist(),
        default=ranked["city"].head(2).tolist(),
        max_selections=3,
    )
    comparison_features = [
        "affordability", "grocery_affordability", "entertainment", "sunset_scenery",
        "safety", "jobs", "internet", "healthcare", "heritage", "nature",
    ]
    columns = st.columns(max(1, len(compare_names)))
    for column, city in zip(columns, compare_names):
        row = df.loc[df["city"] == city].iloc[0]
        with column:
            st.plotly_chart(radar_figure(row, comparison_features), use_container_width=True)
            st.caption(city_description(row))


def render_cities(df: pd.DataFrame, feedback_state: Dict[str, Dict[str, int]]) -> None:
    render_section_head(
        t("city_explorer"),
        t("city_explorer_title"),
        t("city_explorer_sub"),
    )

    search = st.text_input(t("filter_cities"), placeholder="Search by city, region, or tag")
    filtered = df.copy()
    if search.strip():
        needle = search.lower().strip()
        mask = filtered.apply(
            lambda row: needle in " ".join([str(row.get("city", "")), str(row.get("region", "")), str(row.get("tags", ""))]).lower(),
            axis=1,
        )
        filtered = filtered.loc[mask]

    for start in range(0, len(filtered), 3):
        cols = st.columns(3)
        batch = filtered.iloc[start : start + 3]
        for offset, (col, (_, row)) in enumerate(zip(cols, batch.iterrows())):
            with col:
                render_city_card(row)

    st.markdown(f"### {t('profile')}")
    city = st.selectbox(t("select_city"), df["city"].tolist(), key="city_explorer")
    row = df.loc[df["city"] == city].iloc[0]
    left, right = st.columns([1.15, 1])
    with left:
        st.image(city_image_path(city), use_container_width=True, caption=f"{city} city photo")
        st.markdown(f"## {row['city']}")
        st.write(f"**{t('region')}:** {row['region']}")
        st.write(city_description(row))
        st.write(f"**{t('tags')}:** {row['tags']}")
        score_table = pd.DataFrame(
            {
                t("dimension"): [localized_feature_label(feature) for feature in FEATURE_COLUMNS],
                t("score"): [float(row[feature]) for feature in FEATURE_COLUMNS],
            }
        ).sort_values(t("score"), ascending=False)
        st.dataframe(score_table, hide_index=True, use_container_width=True)
    with right:
        st.plotly_chart(
            radar_figure(row, ["affordability", "entertainment", "sunset_scenery", "safety", "jobs", "internet", "healthcare", "heritage", "nature"]),
            use_container_width=True,
        )
    render_feedback_buttons(city, feedback_state, "explorer")


def render_ai_lab(df: pd.DataFrame) -> None:
    render_section_head(
        t("ai_lab"),
        t("ai_title"),
        t("ai_sub"),
    )
    c1, c2, c3, c4 = st.columns(4)
    items = [
        ("01", "Preference parser", "Phrase rules convert natural-language requests into transparent feature weights."),
        ("02", "Machine learning", "K-means groups city profiles into lifestyle archetypes from their feature vectors."),
        ("03", "Deep neural network", "A 64-32-16 MLP estimates user-city like probability using synthetic interactions for the demo."),
        ("04", "Feedback learning", "A Beta-Bernoulli bandit turns likes and dislikes into a small adaptive ranking signal."),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3, c4], items):
        with col:
            st.markdown(
                f'<div class="ai-card"><div class="ai-num">{num}</div><div class="ai-title">{title}</div><div class="ai-desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(f"### {t('data_archetypes')}")
    clustered = cluster_cities(df)
    cluster_view = clustered[["city", "region", "city_archetype"] + FEATURE_COLUMNS].copy()
    cluster_view.columns = ["City", "Region", "Archetype"] + [localized_feature_label(f) for f in FEATURE_COLUMNS]
    st.dataframe(cluster_view, hide_index=True, use_container_width=True)

    st.markdown(f"### {t('team')}")
    st.markdown(
        """
        <div class="advisor-card">
          <b>Author:</b> Akbarxon Nasirov<br>
          <b>Mentor:</b> Dr. Qingyang Xiao<br><br>
          This project is designed as an educational AI portfolio platform that demonstrates data engineering, machine learning, neural networks, feedback learning, geospatial visualization, and Streamlit deployment.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(t("production_warning"))


def render_about() -> None:
    render_section_head(
        t("about"),
        t("about_title"),
        t("about_sub"),
    )
    image = image_data_uri("detail-forest-2.jpg")
    st.markdown(
        f"""
        <div class="hero-shell" style="min-height:430px;">
          <img src="{image}" alt="Lifestyle design visual" style="height:430px;">
          <div class="hero-overlay"></div>
          <div class="hero-copy" style="bottom:38px;">
            <div class="hero-eyebrow">Design + data + AI</div>
            <div class="hero-title" style="font-size:3rem;">Explore before<br>you relocate</div>
            <div class="hero-sub">The experience is inspired by the uploaded UI package and rebuilt in Streamlit so it can deploy directly from GitHub to Streamlit Community Cloud.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown(f"### {t('included')}")
        st.markdown(
            """
            - Natural-language city preference search
            - Uzbekistan map visualization
            - Multi-city scoring and radar comparison
            - ML clustering and neural-network demonstration
            - Like/dislike feedback learning
            - Optional public weather and city context
            - Responsive Streamlit UI styled after the uploaded React/Tailwind design
            """
        )
    with right:
        st.markdown(f"### {t('ui_preserve')}")
        st.write(
            "The GitHub repository also contains the uploaded React/Tailwind UI project under `ui_source/`. "
            "Its `.env` file is intentionally excluded so secrets or environment-specific values are not published."
        )
        st.info(
            "The bundled photos are UI design assets. They are used as lifestyle visuals and should not be treated as verified photographs of specific Uzbekistan cities."
        )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-shell">
          <div class="footer-brand">🧭 Akbarxon AI Living Advisor</div>
          <div class="footer-copy">AI-assisted city discovery for Uzbekistan · Streamlit portfolio prototype<br>Author: Akbarxon Nasirov · Mentor: Dr. Qingyang Xiao</div>
          <div class="footer-rule"></div>
          <div class="footer-copy">Prototype city scores are illustrative, not official statistics. Verify housing, employment, safety, healthcare, visa, legal, and financial information before making relocation decisions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Akbarxon AI Living Advisor",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    df = load_city_data()
    feedback_state = load_feedback_state(df["city"])
    record_app_visit()

    with st.sidebar:
        st.markdown(
            """
            <div class="team-card">
              <div class="team-title">Akbarxon AI Living Advisor</div>
              <div class="team-name"><b>Author:</b> Akbarxon Nasirov<br><b>Mentor:</b> Dr. Qingyang Xiao</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        language_display = st.selectbox(t("language"), list(LANGUAGE_OPTIONS.keys()), index=list(LANGUAGE_OPTIONS.values()).index(st.session_state.get("language", "en")))
        selected_language = LANGUAGE_OPTIONS[language_display]
        if selected_language != st.session_state.get("language", "en"):
            st.session_state["language"] = selected_language
            st.rerun()
        lang = st.session_state.get("language", "en")
        sidebar_count = st.session_state.get("usage_count")
        render_usage_counter(sidebar_count)
        st.markdown(f"#### {t('your_priorities')}")
        st.caption(t("priority_caption"))
        top_n = st.slider(t("num_recs"), 3, 8, 5)
        with st.expander(t("advanced"), expanded=False):
            slider_weights = {
                feature: float(st.slider(localized_feature_label(feature), 0, 5, 0, key=f"slider_{feature}"))
                for feature in FEATURE_COLUMNS
            }
        st.divider()
        st.markdown(
            f"<div class=\"prototype-note\"><b>{html.escape(t('prototype_note'))}</b><br>{html.escape(t('prototype_copy'))}</div>",
            unsafe_allow_html=True,
        )

    render_top_brand()
    nav_labels = {
        "Home": t("nav_home"),
        "Recommend": t("nav_recommend"),
        "Map & Compare": t("nav_map"),
        "Cities": t("nav_cities"),
        "AI Lab": t("nav_ai"),
        "About": t("nav_about"),
    }
    nav_choice = st.radio(
        "Navigation",
        list(nav_labels.values()),
        horizontal=True,
        label_visibility="collapsed",
    )
    page = next(key for key, value in nav_labels.items() if value == nav_choice)
    st.divider()

    # Show the cumulative counter in the main canvas on every navigation page.
    render_usage_counter(st.session_state.get("usage_count"))

    if page == "Home":
        render_home(df, slider_weights, feedback_state, top_n)
    elif page == "Recommend":
        render_recommend(df, slider_weights, feedback_state, top_n)
    elif page == "Map & Compare":
        render_map_compare(df, slider_weights, feedback_state, top_n)
    elif page == "Cities":
        render_cities(df, feedback_state)
    elif page == "AI Lab":
        render_ai_lab(df)
    else:
        render_about()

    render_footer()


if __name__ == "__main__":
    main()
