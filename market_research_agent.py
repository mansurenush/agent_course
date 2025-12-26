# Market Research Agent System
# Using OpenRouter DeepSeek-R1 + Gradio UI

import os
import json
import asyncio
import logging
import time
import random
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

import requests
import time
import random
import re
from typing import List, Dict
from bs4 import BeautifulSoup 


import gradio as gr
from openai import OpenAI
import requests
from duckduckgo_search import DDGS
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
from io import BytesIO

import nest_asyncio
nest_asyncio.apply()



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration & Constants

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "deepseek/deepseek-r1-0528:free"
AGENT_SYSTEM_PROMPT = """Ты - ведущий аналитик рынка. Твоя задача — провести глубокий анализ и написать ДЕТАЛЬНЫЙ отчет.

КРИТИЧЕСКИ ВАЖНО:
1. Твой ответ должен быть ТОЛЬКО валидным JSON объектом.
2. НЕ используй markdown (``````).
3. Начинай ответ сразу с { и заканчивай на }.
4. Каждый раздел должен содержать минимум 3-4 предложения с конкретными данными и цитатами источников.
5. После каждого факта указывай номер источника в квадратных скобках: [1], [2], [15] и т.д.

Пример корректного формата:
{
  "Обзор рынка": [
    "Текст с данными [1].",
    "Еще факт [3][5]."
  ]
}

НЕ ПИШИ ничего кроме JSON объекта."""


@dataclass
class MarketResearchRequest:
    """Структура запроса на анализ рынка"""
    topic: str
    structure: Dict[str, Any]  
    include_competitors: bool = True
    num_competitors: int = 5
    include_trends: bool = True
    include_risks: bool = True

@dataclass
class ResearchResult:
    """Результаты исследования рынка"""
    topic: str
    research_data: Dict[str, Any]
    web_sources: List[str]
    timestamp: str

# Web Search Module

class WebSearchEngine:
    """Поиск через SerpWow API (Google SERP в JSON)"""

    def __init__(self):
        self.api_key = os.getenv("SERPWOW_API_KEY", "")
        self.session = requests.Session()

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        try:
            params = {
                "api_key": self.api_key,
                "q": query,
                "location": "Moscow,Russia",
                "gl": "ru",
                "hl": "ru",
                "google_domain": "google.ru",
                "engine": "google",
            }
            r = self.session.get("https://api.serpwow.com/search", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            organic = data.get("organic_results", [])[:max_results]
            results = []
            for item in organic:
                results.append({
                    "title": item.get("title", ""),
                    "href": item.get("link", ""),
                    "body": item.get("snippet", ""),
                })
            logger.info(f"SerpWow search OK: {len(results)} results for '{query}'")
            return results

        except Exception as e:
            logger.error(f"SerpWow search error for '{query}': {e}")
            return []
        
    @staticmethod
    def extract_text_from_search(results: List[Dict[str, str]]) -> str:
        """
        Приводит результаты (title/href/body) к текстовому контексту для LLM.
        Совместимо с duckduckgo_search и SerpWow, т.к. используем те же ключи.
        """
        if not results:
            return "Информация не найдена."

        parts = []
        for r in results:
            title = r.get("title", "N/A")
            href = r.get("href", "")
            body = r.get("body", "")
            if body:
                parts.append(f"- {title}: {body} ({href})")
            else:
                parts.append(f"- {title} ({href})")
        return "\n".join(parts)

# Agent Module

class MarketResearchAgent:
    """Основной агент для анализа рынка"""
    
    def __init__(self):
        """Инициализация агента с OpenRouter client"""
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL
        )
        self.conversation_history = []
        self.research_sources = []
        self.search_engine = WebSearchEngine()  
    
    def reset_conversation(self):
        """Сброс истории разговора"""
        self.conversation_history = []
        self.research_sources = []
    
    def add_message(self, role: str, content: str):
        """Добавление сообщения в историю"""
        self.conversation_history.append({"role": role, "content": content})
    
    def search_market_info(self, topic: str, queries: Optional[List[str]] = None) -> Dict[str, Any]:
        logger.info(f"Searching market info for: {topic}")
        
        if queries is None: 
            queries = [
                f'Top companies in "{topic}" market 2025 comparison',
                f'"{topic}" market size share statistics 2024 2025',
                f'Future trends and challenges in "{topic}" industry 2025'
            ]

        collected_info = {}
        
        for query in queries:
            results = self.search_engine.search(query, max_results=10)
            
            if results:
                extracted_text = self.search_engine.extract_text_from_search(results)
                collected_info[query] = {
                    "results_count": len(results),
                    "summary": extracted_text
                }
                for result in results:
                    href = result.get('href', '')
                    if href and href not in self.research_sources:
                        self.research_sources.append(href)
                        logger.info(f"Added source: {href}")  # Для отладки
            else:
                collected_info[query] = {
                    "results_count": 0,
                    "summary": "Поиск не дал результатов (попробуйте другую тему)"
                }

        logger.info(f"Total sources collected: {len(self.research_sources)}")

        return collected_info
    
    def analyze_market(self, topic: str, structure: Dict[str, Any], 
                      collected_info: Dict[str, Any]) -> str:
        """
        Анализ рынка с помощью DeepSeek R1
        """
        logger.info(f"Analyzing market for: {topic}")
        
        # Формирование контекста поиска
        search_context = "СОБРАННАЯ ИНФОРМАЦИЯ ИЗ ИНТЕРНЕТА:\n"
        for query, data in collected_info.items():
            search_context += f"\nПоиск: '{query}'\n"
            search_context += data["summary"] + "\n"
        
        sources_list = "\n\nИСПОЛЬЗУЙ ЭТИ ИСТОЧНИКИ ДЛЯ ЦИТИРОВАНИЯ:\n"
        for i, url in enumerate(self.research_sources, 1):
            sources_list += f"[{i}] {url}\n"

        # Первое сообщение - контекст и задача
        user_message = f"""
            Проведи анализ рынка по теме: {topic}

            {search_context}

            {sources_list}

            Структура отчета, которую нужно соблюдать:
            {json.dumps(structure, ensure_ascii=False, indent=2)}

            ТРЕБОВАНИЯ:
            1. Используй данные из источников
            2. Цитируй источники после каждого факта: [1], [2], [15]
            3. Минимум 3-4 предложения на раздел
            4. Если есть цифры по годам - добавь "chart_data": {{"title": "...", "labels": [...], "values": [...]}}

            ФОРМАТ: Верни ТОЛЬКО JSON объект, начинающийся с {{ и заканчивающийся на }}. Никакого текста до или после.
            """
        self.add_message("user", user_message)
        
        # Отправка запроса к DeepSeek R1
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    *self.conversation_history
                ],
                temperature=0.7,
                max_tokens=20000,
                top_p=0.95
            )
            
            result = response.choices[0].message.content

            logger.info(f"Response length: {len(result)} chars")
            logger.info(f"Response preview: {result[:200]}...")
            logger.info(f"Response ends with: ...{result[-100:]}")

            self.add_message("assistant", result)
            
            logger.info("Market analysis completed successfully")
            return result
        
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise
    
    def refine_analysis(self, feedback: str) -> str:
        """
        Уточнение анализа на основе обратной связи
        """
        logger.info("Refining analysis based on feedback")
        
        self.add_message("user", feedback)
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    *self.conversation_history
                ],
                temperature=0.7,
                max_tokens=10000
            )
            
            result = response.choices[0].message.content
            self.add_message("assistant", result)
            return result
        
        except Exception as e:
            logger.error(f"Error during refinement: {e}")
            raise

# PDF Report Generation

class PDFReportGenerator:
    """Генератор PDF отчетов с ГАРАНТИРОВАННОЙ поддержкой кириллицы"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.cyrillic_font = self._register_cyrillic_font()
        self._setup_custom_styles()
    
    def _register_cyrillic_font(self) -> str:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import os
        import matplotlib

        mpl_data = matplotlib.get_data_path()
        fonts_dir = os.path.join(mpl_data, "fonts", "ttf")

        regular = os.path.join(fonts_dir, "DejaVuSans.ttf")
        bold = os.path.join(fonts_dir, "DejaVuSans-Bold.ttf")
        oblique = os.path.join(fonts_dir, "DejaVuSans-Oblique.ttf")
        bold_oblique = os.path.join(fonts_dir, "DejaVuSans-BoldOblique.ttf")

        pdfmetrics.registerFont(TTFont("DejaVuSans", regular))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bold))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", oblique))
        pdfmetrics.registerFont(TTFont("DejaVuSans-BoldOblique", bold_oblique))

        pdfmetrics.registerFontFamily(
            "DejaVuSans",
            normal="DejaVuSans",
            bold="DejaVuSans-Bold",
            italic="DejaVuSans-Oblique",
            boldItalic="DejaVuSans-BoldOblique",
        )

        logger.info(f"✅ Using DejaVuSans from matplotlib: {regular}")
        return "DejaVuSans"

    
    def _setup_custom_styles(self):
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import os
        import matplotlib

        mpl_data = matplotlib.get_data_path()
        regular_path = os.path.join(mpl_data, "fonts", "ttf", "DejaVuSans.ttf")
        bold_path    = os.path.join(mpl_data, "fonts", "ttf", "DejaVuSans-Bold.ttf")

        pdfmetrics.registerFont(TTFont("DejaVuSans", regular_path))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bold_path))

        title_font = "DejaVuSans-Bold"
        heading_font = "DejaVuSans-Bold"
        body_font = "DejaVuSans"

        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontName=title_font,
            fontSize=24,
            textColor=colors.HexColor('#1a365d'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontName=heading_font,
            fontSize=14,
            textColor=colors.HexColor('#2d3748'),
            spaceAfter=12,
            spaceBefore=12
        ))

        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontName=body_font,
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        ))

        self.styles.add(ParagraphStyle(
            name="CustomMeta",
            parent=self.styles["BodyText"],
            fontName="DejaVuSans-Oblique",
            fontSize=10,
            textColor=colors.HexColor("#4a5568"),
            spaceAfter=12,
        ))

    
    def create_report(self, filename: str, title: str, sections: Dict[str, Any],
                     charts: Optional[Dict[str, Any]] = None,
                     sources: Optional[List[str]] = None) -> str:
        """Создание PDF с КИРИЛЛИЦЕЙ и ССЫЛКАМИ"""
        logger.info(f"Creating PDF report: {filename}")
        
        doc = SimpleDocTemplate(filename, pagesize=A4, encoding='utf-8')
        story = []
        
        # Заголовок
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Дата
        date_str = datetime.now().strftime("%d.%m.%Y")
        story.append(Paragraph(f"Дата отчета: {date_str}", self.styles["CustomMeta"]))
        story.append(Spacer(1, 0.3 * inch))
        
        # Основные разделы
        for section_name, section_content in sections.items():
            # Заголовок раздела
            story.append(Paragraph(section_name, self.styles['CustomHeading']))
            
            if isinstance(section_content, list):
                for item in section_content:

                    if sources:
                        item = self._make_citations_clickable(str(item), sources)

                    formatted_item = item.replace("**", "<b>").replace("**", "</b>")
                    story.append(Paragraph(f"• {formatted_item}", self.styles['CustomBody']))
            elif isinstance(section_content, dict):
                for key, value in section_content.items():
                    if sources:
                        value = self._make_citations_clickable(str(value), sources)

                    story.append(Paragraph(f"<b>{key}:</b> {value}", self.styles['CustomBody']))
            else:
                story.append(Paragraph(str(section_content), self.styles['CustomBody']))
            
            story.append(Spacer(1, 0.1 * inch))
        
        # Графики 
        if charts:
            story.append(PageBreak())
            story.append(Paragraph("", self.styles['CustomHeading']))
            story.append(Spacer(1, 0.1 * inch))
            
            for chart_name, chart_data in charts.items():
                chart_image = self._generate_chart(chart_data)
                if chart_image:
                    # Заголовок графика из данных
                    chart_title = chart_data.get('title', chart_name)
                    story.append(Paragraph(chart_title, self.styles['CustomHeading']))
                    
                    img = RLImage(chart_image, width=6*inch, height=3.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.3 * inch))
        
        # Блок Источников 
        if sources:
            story.append(PageBreak())
            story.append(Paragraph("Использованные источники", self.styles['CustomHeading']))
            story.append(Spacer(1, 0.1 * inch))
            
            for i, url in enumerate(sources, 1):
                display_url = url[:80] + "..." if len(url) > 80 else url
                link_text = f'[{i}] <a href="{url}" color="blue"><u>{display_url}</u></a>'
                story.append(Paragraph(link_text, self.styles['CustomBody']))
                story.append(Spacer(1, 0.05 * inch))

        # Построение
        try:
            doc.build(story)
            logger.info(f"✅ PDF CREATED: {filename}")
            return filename
        except Exception as e:
            logger.error(f"PDF build error: {e}")
            raise
    
    @staticmethod
    def _generate_chart(chart_data: Dict[str, Any]) -> Optional[BytesIO]:
        """Графики с кириллицей"""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import os
            mpl_data = matplotlib.get_data_path()
            dejavu = os.path.join(mpl_data, "fonts", "ttf", "DejaVuSans.ttf")

            plt.rcParams['font.family'] = "DejaVu Sans"
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
            plt.rcParams['font.size'] = 10
            
            chart_type = chart_data.get('type', 'bar')
            labels = chart_data.get('labels', [])
            values = chart_data.get('values', [])
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            if chart_type == 'bar':
                bars = ax.bar(labels, values, color='steelblue', alpha=0.7)
                ax.set_ylabel('Значение')
            elif chart_type == 'line':
                ax.plot(labels, values, marker='o', linewidth=2, color='steelblue')
                ax.set_ylabel('Значение')
            elif chart_type == 'pie':
                ax.pie(values, labels=labels, autopct='%1.1f%%')
            
            ax.set_title(chart_data.get('title', 'График'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return buf
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return None
        
    @staticmethod
    def _make_citations_clickable(text: str, sources: List[str]) -> str:
        """
        Превращает [1], [2] в кликабельные ссылки на источники.
        """
        
        def replace_citation(match):
            num = int(match.group(1))
            # Проверяем, что номер в диапазоне
            if 1 <= num <= len(sources):
                url = sources[num - 1]
                # Создаем кликабельную ссылку синего цвета
                return f'<a href="{url}" color="blue"><u>[{num}]</u></a>'
            return match.group(0)  # Если номер некорректный, оставляем как есть
        
        # Находим все [число] и заменяем их
        return re.sub(r'\[(\d+)\]', replace_citation, text)

# Gradio Interface

class MarketResearchUI:
    """UI интерфейс на Gradio"""
    
    def __init__(self):
        self.agent = MarketResearchAgent()
        self.pdf_generator = PDFReportGenerator()
        self.current_research = None
        self.search_engine = None
    def parse_analysis_result(self, result_text: str) -> Dict[str, Any]:
        """
        Усиленный парсер: извлекает JSON из любого формата ответа.
        """
        
        logger.info(f"Parsing response, length: {len(result_text)} chars")
        
        # Шаг 1: Очистка от пробелов
        clean_text = result_text.strip()
        
        # Шаг 2: Удаление всех markdown блоков
        # Убираем ``````
        # clean_text = re.sub(r'```')
        lean_text = re.sub(r'```json', '', clean_text)
        clean_text = re.sub(r'```', '', clean_text)
        
        # Шаг 3: Поиск JSON объекта (от первой { до последней })
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            logger.info(f"Found JSON candidate, length: {len(json_str)}")
        else:
            logger.error("No JSON structure found in response")
            return {"analysis_text": result_text}
        
        # Шаг 4: Попытка парсинга
        try:
            parsed = json.loads(json_str)
            logger.info("✅ JSON parsed successfully")
            return parsed
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error at position {e.pos}: {e.msg}")
            
            # Шаг 5: Если JSON обрезан, пытаемся закрыть структуру
            try:
                # Считаем открытые скобки
                open_braces = json_str.count('{') - json_str.count('}')
                open_brackets = json_str.count('[') - json_str.count(']')
                
                fixed_json = json_str
                
                # Закрываем незакрытые строки
                if fixed_json.count('"') % 2 != 0:
                    fixed_json += '"'
                
                # Закрываем массивы
                fixed_json += ']' * open_brackets
                
                # Закрываем объекты
                fixed_json += '}' * open_braces
                
                logger.info(f"Trying to fix JSON with {open_braces} braces and {open_brackets} brackets")
                
                parsed = json.loads(fixed_json)
                logger.info("✅ JSON fixed and parsed successfully")
                return parsed
            
            except Exception:
                pass
            
            logger.error("All parsing attempts failed, returning raw text")
            return {"analysis_text": result_text}

    
    def conduct_research(self, topic: str, structure_text: str, 
                        progress=gr.Progress()) -> tuple:
        """
        Проведение исследования рынка
        """
        try:
            progress(0, desc="Инициализация...")
            
            # Парсинг структуры отчета
            try:
                structure = json.loads(structure_text) if structure_text.strip() else {
                    "Обзор рынка": [],
                    "Анализ конкурентов": [],
                    "Ключевые тренды": [],
                    "Возможности и риски": []
                }
            except json.JSONDecodeError:
                structure = {
                    "Обзор рынка": [],
                    "Анализ конкурентов": [],
                    "Ключевые тренды": [],
                    "Возможности и риски": []
                }
            
            # Сброс предыдущей сессии
            self.agent.reset_conversation()
            progress(0.2, desc="Поиск информации в интернете...")
            
            # Поиск информации
            collected_info = self.agent.search_market_info(topic)

            logger.info(f"Sources after search: {len(self.agent.research_sources)}")

            progress(0.5, desc="Анализ данных с DeepSeek R1...")
            
            # Анализ
            analysis_result = self.agent.analyze_market(topic, structure, collected_info)
            progress(0.8, desc="Обработка результатов...")
            
            # Парсинг результата
            parsed_result = self.parse_analysis_result(analysis_result)
            
            # Сохранение для последующего использования
            self.current_research = ResearchResult(
                topic=topic,
                research_data=parsed_result,
                web_sources=self.agent.research_sources,
                timestamp=datetime.now().isoformat()
            )
            
            progress(1.0, desc="Готово!")
            
            # Форматирование вывода
            formatted_output = self._format_research_output(parsed_result)
            # sources_text = "Sources:\n" + "\n".join([f"- {s}" for s in self.agent.research_sources[:10]])
            if self.agent.research_sources:
                sources_text = "Использованные источники:\n" + "\n".join(
                    [f"[{i}] {s}" for i, s in enumerate(self.agent.research_sources, 1)]
                )
            else:
                sources_text = "Источники не найдены (проверьте API поиска)"

            return formatted_output, sources_text, parsed_result
        
        except Exception as e:
            logger.error(f"Research error: {e}")
            return f"Error: {str(e)}", "", {}
    
    def _format_research_output(self, data: Dict[str, Any]) -> str:
        """
        Форматирование вывода для отображения
        """
        if isinstance(data, dict):
            if len(data) == 1 and "analysis_text" in data:
                return "⚠️ ОШИБКА ПАРСИНГА JSON\n\n" + data["analysis_text"]

            output = ""
            for key, value in data.items():
                if key in ["chart_data", "analysis_text"]:
                    continue

                output += f"\n{'='*60}\n"
                output += f"{key}\n"
                output += f"{'='*60}\n"
                if isinstance(value, list):
                    for item in value:
                        clean_item = str(item).replace("**", "")
                        output += f"• {item}\n"
                elif isinstance(value, dict):
                    for k, v in value.items():
                        output += f"{k}: {v}\n"
                else:
                    output += str(value) + "\n"
            return output
        return str(data)
    
    def generate_pdf_report(self, pdf_filename: str = None, 
                          progress=gr.Progress()) -> tuple:
        """
        Генерация PDF отчета (ИСПРАВЛЕННАЯ ВЕРСИЯ)
        """
        try:
            if not self.current_research:
                return "No research data available. Conduct research first.", ""
            
            progress(0.2, desc="Подготовка данных...")
            
            if not pdf_filename:
                # Генерируем имя файла с датой
                sanitized_topic = "".join([c for c in self.current_research.topic if c.isalnum() or c in (' ', '-', '_')]).strip()
                pdf_filename = f"report_{sanitized_topic}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            
            # --- ИСПРАВЛЕНИЕ 1: Фильтрация данных ---
            sections = {}
            charts = {}
            
            # Разбираем JSON от LLM
            raw_data = self.current_research.research_data
            
            # 1. Ищем данные для графика
            if "chart_data" in raw_data:
                c_data = raw_data["chart_data"]
                if isinstance(c_data, dict) and c_data.get("values"):
                    charts["Аналитический график"] = {
                        "type": "bar",
                        "title": c_data.get("title", "Динамика показателей"),
                        "labels": c_data.get("labels", []),
                        "values": c_data.get("values", [])
                    }
            
            # 2.  текстовые разделы
            for key, value in raw_data.items():
                if key == "chart_data":
                    continue 
                
                # Сохраняем остальные разделы
                if isinstance(value, list):
                    sections[key] = value
                elif isinstance(value, dict):
                    sections[key] = value
                else:
                    sections[key] = [str(value)]
            
            progress(0.6, desc="Генерация PDF...")
            
            # Создание отчета с передачей источников
            pdf_path = self.pdf_generator.create_report(
                filename=pdf_filename,
                title=f"Анализ рынка: {self.current_research.topic}",
                sections=sections,
                charts=charts,
                sources=self.current_research.web_sources 
            )
            
            progress(1.0, desc="Отчет создан!")
            return f"PDF Report created: {pdf_path}", pdf_path
        
        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            return f"Error generating PDF: {str(e)}", ""
    
    def refine_research(self, feedback: str, progress=gr.Progress()) -> str:
        """
        Уточнение исследования
        """
        try:
            progress(0.3, desc="Обработка обратной связи...")
            refinement = self.agent.refine_analysis(feedback)
            progress(1.0, desc="Уточнение завершено!")
            return refinement
        except Exception as e:
            logger.error(f"Refinement error: {e}")
            return f"Error: {str(e)}"

# Main Gradio Interface

def create_gradio_interface():
    """
    Создание основного интерфейса Gradio
    """
    ui = MarketResearchUI()
    
    with gr.Blocks(title="Market Research Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 Market Research Agent System
        
        Автономная система для проведения анализа рынка по заданной тематике.
        Использует AI (DeepSeek R1) для глубокого анализа и поиск информации из интернета.
        """)
        
        with gr.Tabs():
            # TAB 1: Research Execution
            with gr.Tab("📊 Исследование"):
                gr.Markdown("### Параметры исследования")
                
                with gr.Row():
                    topic_input = gr.Textbox(
                        label="Тема для анализа",
                        placeholder="Например: электромобили, IoT резиновые женщины, сигареты с протеином...",
                        lines=2
                    )
                
                with gr.Row():
                    structure_input = gr.Textbox(
                        label="Структура отчета (JSON, опционально)",
                        placeholder='{"Обзор рынка": [], "Конкуренты": [], ...}',
                        lines=5,
                        value='{\n  "Обзор рынка": [],\n  "Анализ конкурентов": [],\n  "Ключевые тренды": [],\n  "Возможности и риски": []\n}'
                    )
                
                research_button = gr.Button("🚀 Начать исследование", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        analysis_output = gr.Textbox(
                            label="Результаты анализа",
                            lines=15,
                            interactive=False
                        )
                    with gr.Column():
                        sources_output = gr.Textbox(
                            label="Использованные источники",
                            lines=15,
                            interactive=False
                        )
                
                # Hidden state для хранения данных
                research_state = gr.State()
                
                # Функция обработки исследования
                def run_research(topic, structure):
                    result, sources, state = ui.conduct_research(topic, structure)
                    return result, sources, state
                
                research_button.click(
                    fn=run_research,
                    inputs=[topic_input, structure_input],
                    outputs=[analysis_output, sources_output, research_state]
                )
            
            # TAB 2: Report Generation
            with gr.Tab("📄 Генерация отчета"):
                gr.Markdown("### Создание PDF отчета")
                
                with gr.Row():
                    pdf_filename_input = gr.Textbox(
                        label="Имя файла отчета (опционально)",
                        placeholder="market_report.pdf"
                    )
                
                pdf_button = gr.Button("📥 Создать PDF отчет", variant="primary", size="lg")
                
                with gr.Row():
                    pdf_status = gr.Textbox(
                        label="Статус",
                        interactive=False
                    )
                    pdf_download = gr.File(
                        label="Скачать отчет",
                        interactive=False
                    )
                
                pdf_button.click(
                    fn=ui.generate_pdf_report,
                    inputs=[pdf_filename_input],
                    outputs=[pdf_status, pdf_download]
                )
            
            # TAB 3: Refinement
            with gr.Tab("🔧 Уточнение анализа"):
                gr.Markdown("### Уточнение результатов исследования")
                
                feedback_input = gr.Textbox(
                    label="Ваша обратная связь",
                    placeholder="Например: добавьте информацию о..., уточните...",
                    lines=5
                )
                
                refine_button = gr.Button("✅ Уточнить анализ", variant="primary")
                
                refinement_output = gr.Textbox(
                    label="Уточненный результат",
                    lines=10,
                    interactive=False
                )
                
                refine_button.click(
                    fn=ui.refine_research,
                    inputs=[feedback_input],
                    outputs=[refinement_output]
                )
            
            # TAB 4: Info
            with gr.Tab("ℹ️ Информация"):
                gr.Markdown("""
                ## О системе
                
                Это агентская система для автономного анализа рынка, построенная на:
                
                ### Компоненты:
                - **AI Модель**: DeepSeek R1 через OpenRouter
                - **Поиск информации**: DuckDuckGo Search
                - **UI**: Gradio
                - **Отчеты**: PDF с графиками и таблицами
                
                ### Возможности:
                ✓ Автоматический поиск информации в интернете
                ✓ Глубокий анализ с использованием DeepSeek R1
                ✓ Генерация структурированных PDF отчетов
                ✓ Поддержка пользовательских структур отчетов
                ✓ Уточнение результатов на основе обратной связи
                
                ### Использование:
                1. Введите тему для анализа
                2. (Опционально) задайте структуру отчета в JSON
                3. Запустите исследование
                4. Просмотрите результаты
                5. Создайте PDF отчет
                6. Уточните результаты если нужно
                
                ### Требования:
                - Интернет соединение
                - OpenRouter API ключ
                """)
    
    return demo


if __name__ == "__main__":
    logger.info("Starting Market Research Agent System...")
    
    # Проверка API ключа
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.startswith('sk-or-v1-cffa2e71'):
        logger.warning("Using default OpenRouter API key. For production, set OPENROUTER_API_KEY env variable.")
    
    # Создание и запуск интерфейса
    demo = create_gradio_interface()
    
    logger.info("Launching Gradio interface...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7875,
        share=False,
        show_error=True
    )
