import re
import PyPDF2
import io
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
import json
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class ResumeAnalysisAgent:
    def __init__(self, api_key, cutoff_score=75, model=None):
        self.api_key = api_key
        self.cutoff_score = cutoff_score
        self.model = model or DEFAULT_MODEL
        self.resume_text = None
        self.rag_vectorstore = None
        self.analysis_result = None
        self.jd_text = None
        self.extracted_skills = None
        self.resume_weaknesses = []
        self.resume_strengths = []
        self.improvement_suggestions = {}

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a PDF file"""
        try:
            if hasattr(pdf_file, "getvalue"):
                pdf_data = pdf_file.getvalue()
                pdf_file_like = io.BytesIO(pdf_data)
                reader = PyPDF2.PdfReader(pdf_file_like)
            else:
                reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page in reader.pages:
                text += page.extract_text()

            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def extract_text_from_txt(self, txt_file):
        """Extract text from a text file"""
        try:
            if hasattr(txt_file, "getvalue"):
                return txt_file.getvalue().decode("utf-8")
            else:
                with open(txt_file, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            print(f"Error extracting text from text file: {e}")
            return ""

    def extract_text_from_file(self, file):
        """Extract text from a file (PDF or TXT)"""
        if hasattr(file, "name"):
            file_extension = file.name.split(".")[-1].lower()
        else:
            file_extension = file.split(".")[-1].lower()

        if file_extension == "pdf":
            return self.extract_text_from_pdf(file)
        elif file_extension == "txt":
            return self.extract_text_from_txt(file)
        else:
            print(f"Unsupported file extension: {file_extension}")

    def create_rag_vector_store(self, text):
        """Create a vector store for RAG"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore

    def create_vector_store(self, text):
        """Create a simpler vector store for skill analysis"""
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        vectorstore = FAISS.from_texts([text], embeddings)
        return vectorstore

    def build_retrieval_chain(self, retriever, model=None, extra_context=""):
        """Build an LCEL retrieval chain: invoke with {"input": ...} -> {"answer": ...}

        `extra_context` is static text prepended to the retrieved documents —
        used by Q&A to supply the analysis, which is not in the vector store.
        """
        if extra_context:
            # Braces in the text would be read as template variables.
            safe = extra_context.replace("{", "{{").replace("}", "}}")
            template = (
                "You are helping a candidate understand their own resume.\n"
                "Answer using the resume analysis and the resume excerpts below.\n"
                "The analysis is authoritative for scores, strengths and weaknesses — "
                "when asked about weaknesses, gaps or areas to improve, answer from it "
                "and never claim the information is unavailable.\n"
                "Quote concrete scores where they help. If something genuinely is not "
                "covered by either source, say so briefly and answer what you can.\n\n"
                f"Resume analysis:\n{safe}\n\n"
                "Resume excerpts:\n{context}\n\n"
                "Question: {input}"
            )
        else:
            template = (
                "Answer the question using only the context below.\n\n"
                "Context:\n{context}\n\n"
                "Question: {input}"
            )
        prompt = ChatPromptTemplate.from_template(template)
        combine_docs_chain = create_stuff_documents_chain(
            ChatOpenAI(model=model or self.model, api_key=self.api_key), prompt
        )
        return create_retrieval_chain(retriever, combine_docs_chain)

    def analyze_skill(self, qa_chain, skill):
        """Analyze a skill in the resume"""
        query = f"On a scale of 0-10, how clearly does the candidate mention proficiency in {skill}? Provide a numeric rating first, followed by reasoning."
        response = qa_chain.invoke({"input": query})["answer"]
        match = re.search(r"(\d{1,2})", response)
        score = int(match.group(1)) if match else 0

        reasoning = (
            response.split(".", 1)[1].strip()
            if "." in response and len(response.split(".")) > 1
            else ""
        )

        return skill, min(score, 10), reasoning

    def analyze_resume_weaknesses(self):
        """Analyze specific weaknesses in the resume based on missing skills"""
        if (
            not self.resume_text
            or not self.extracted_skills
            or not self.analysis_result
        ):
            return []

        weaknesses = []

        for skill in self.analysis_result.get("missing_skills", []):

            llm = ChatOpenAI(model=self.model, api_key=self.api_key)
            prompt = f"""
                Analyze why the resume is weak in demonstrating proficiency in "{skill}".

                For your analysis, consider:
                1. What's missing from the resume regarding this skill?
                2. How could it be improved with specific examples?
                3. What specific action items would make this skill stand out?

                Resume Content:
                {self.resume_text[:3000]}...

                Provide your response in this JSON format:
                {{
                    "weakness": "A concise description of what's missing or problematic (1-2 sentences)",
                    "improvement_suggestions": [
                        "Specific suggestion 1",
                        "Specific suggestion 2",
                        "Specific suggestion 3"
                    ],
                    "example_addition": "A specific bullet point that could be added to showcase this skill"
                }}

                Return only valid JSON, no other text.
                """

            response = llm.invoke(prompt)
            weakness_content = response.content.strip()

            try:
                weakness_data = json.loads(weakness_content)

                weakness_detail = {
                    "skill": skill,
                    "score": self.analysis_result.get("skill_scores", {}).get(skill, 0),
                    "detail": weakness_data.get(
                        "weakness", "No specific details provided."
                    ),
                    "suggestions": weakness_data.get("improvement_suggestions", []),
                    "example": weakness_data.get("example_addition", ""),
                }

                weaknesses.append(weakness_detail)

                self.improvement_suggestions[skill] = {
                    "suggestions": weakness_data.get("improvement_suggestions", []),
                    "example": weakness_data.get("example_addition", ""),
                }
            except json.JSONDecodeError:

                weaknesses.append(
                    {
                        "skill": skill,
                        "score": self.analysis_result.get("skill_scores", {}).get(
                            skill, 0
                        ),
                        "detail": weakness_content[
                            :200
                        ],  # Truncate if it's not proper JSON
                    }
                )

        self.resume_weaknesses = weaknesses
        return weaknesses

    def summarize_job_description(self, jd_text):
        """Pull the headline facts out of a job description.

        Returns company, title, experience, employment type, key skills and
        nice-to-haves. Fields the posting doesn't state come back empty rather
        than invented.
        """
        try:
            llm = ChatOpenAI(model=self.model, api_key=self.api_key)
            prompt = f"""
                Read this job description and extract only what it actually states.

                Return ONLY a JSON object, no prose and no markdown fences:
                {{
                  "company": "hiring company, or \"\" if not named",
                  "title": "the role title",
                  "experience": "years of experience asked for, e.g. \"5+ Years\", or \"\"",
                  "employment_type": "Full-time, Contract, Internship etc., or \"\"",
                  "key_skills": ["required skills, at most 6"],
                  "nice_to_have": ["preferred but not required, at most 6"]
                }}

                Do not guess. If the posting does not state something, use an
                empty string or an empty list.

                Job Description:
                {jd_text[:6000]}
                """

            raw = self._strip_json_fence(llm.invoke(prompt).content)

            parsed = json.loads(raw)
            return {
                "company": str(parsed.get("company") or ""),
                "title": str(parsed.get("title") or ""),
                "experience": str(parsed.get("experience") or ""),
                "employment_type": str(parsed.get("employment_type") or ""),
                "key_skills": [str(x) for x in (parsed.get("key_skills") or [])][:6],
                "nice_to_have": [str(x) for x in (parsed.get("nice_to_have") or [])][:6],
            }
        except Exception as e:
            print(f"Could not summarize job description: {e}")
            return {
                "company": "",
                "title": "",
                "experience": "",
                "employment_type": "",
                "key_skills": [],
                "nice_to_have": [],
            }

    def extract_skills_from_jd(self, jd_text):
        """Extract skills from a job description"""
        try:
            llm = ChatOpenAI(model=self.model, api_key=self.api_key)
            prompt = f"""
                Extract a comprehensive list of technical skills, technologies, and competencies required from this job description.
                Format the output as a Python list of strings. Only include the list, nothing else.

                Job Description:
                {jd_text}
                """

            response = llm.invoke(prompt)
            skills_text = response.content

            match = re.search(r"\[(.*?)\]", skills_text, re.DOTALL)
            if match:
                skills_text = match.group(0)

            try:
                skills_list = eval(skills_text)
                if isinstance(skills_list, list):
                    return skills_list
            except:
                pass

            skills = []
            for line in skills_text.split("\n"):
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    skill = line[2:].strip()
                    if skill:
                        skills.append(skill)
                elif line.startswith('"') and line.endswith('"'):
                    skill = line.strip('"')
                    if skill:
                        skills.append(skill)

            return skills
        except Exception as e:
            print(f"Error extracting skills from job description: {e}")
            return []

    def semantic_skill_analysis(self, resume_text, skills):
        """Analyze skills semantically"""
        vectorstore = self.create_vector_store(resume_text)
        retriever = vectorstore.as_retriever()
        qa_chain = self.build_retrieval_chain(retriever)

        skill_scores = {}
        skill_reasoning = {}
        missing_skills = []
        total_score = 0

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(
                executor.map(lambda skill: self.analyze_skill(qa_chain, skill), skills)
            )

        for skill, score, reasoning in results:
            skill_scores[skill] = score
            skill_reasoning[skill] = reasoning
            total_score += score
            if score <= 5:
                missing_skills.append(skill)

        overall_score = int((total_score / (10 * len(skills))) * 100)
        selected = overall_score >= self.cutoff_score

        reasoning = "Candidate evaluated based on explicit resume content using semantic similarity and clear numeric scoring."
        strengths = [skill for skill, score in skill_scores.items() if score >= 7]
        improvement_areas = missing_skills if not selected else []

        self.resume_strengths = strengths

        return {
            "overall_score": overall_score,
            "skill_scores": skill_scores,
            "skill_reasoning": skill_reasoning,
            "selected": selected,
            "reasoning": reasoning,
            "missing_skills": missing_skills,
            "strengths": strengths,
            "improvement_areas": improvement_areas,
        }

    def analyze_resume(self, resume_file, role_requirements=None, custom_jd=None):
        """Analyze a resume against role requirements or a custom JD"""
        self.resume_text = self.extract_text_from_file(resume_file)

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(self.resume_text)
            self.resume_file_path = tmp.name

        self.rag_vectorstore = self.create_rag_vector_store(self.resume_text)

        if custom_jd:
            self.jd_text = self.extract_text_from_file(custom_jd)
            self.extracted_skills = self.extract_skills_from_jd(self.jd_text)
            self.analysis_result = self.semantic_skill_analysis(
                self.resume_text, self.extracted_skills
            )

        elif role_requirements:
            self.extracted_skills = role_requirements
            self.analysis_result = self.semantic_skill_analysis(
                self.resume_text, role_requirements
            )

        if (
            self.analysis_result
            and "missing_skills" in self.analysis_result
            and self.analysis_result["missing_skills"]
        ):
            self.analyze_resume_weaknesses()
            self.analysis_result["detailed_weaknesses"] = self.resume_weaknesses

        return self.analysis_result

    def _analysis_context(self):
        """The analysis as plain text.

        The vector store only holds resume chunks, so without this the model
        has no idea what its own scoring found and answers questions about
        weaknesses with "the context doesn't mention any".
        """
        result = self.analysis_result or {}
        if not result:
            return ""

        lines = []
        score = result.get("overall_score")
        if score is not None:
            verdict = "at or above" if result.get("selected") else "below"
            lines.append(
                f"Overall ATS score: {score}/100 ({verdict} the cutoff of {self.cutoff_score})."
            )

        scores = result.get("skill_scores") or {}
        if scores:
            ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            lines.append(
                "Skill scores out of 10: "
                + ", ".join(f"{name} {value}/10" for name, value in ordered)
            )

        strengths = result.get("strengths") or []
        if strengths:
            lines.append("Strengths (scored 7 or above): " + ", ".join(strengths))

        missing = result.get("missing_skills") or []
        if missing:
            lines.append(
                "Weaknesses and gaps (scored 5 or below): " + ", ".join(missing)
            )
        else:
            lines.append("No skill scored 5 or below, so there are no flagged gaps.")

        for weakness in result.get("detailed_weaknesses") or []:
            lines.append(
                f"Weakness detail - {weakness.get('skill', '')} "
                f"({weakness.get('score', 0)}/10): {weakness.get('detail', '')}"
            )
            for suggestion in weakness.get("suggestions") or []:
                lines.append(f"  Suggestion: {suggestion}")

        if self.extracted_skills:
            lines.append(
                "Skills the target role requires: " + ", ".join(self.extracted_skills)
            )

        return "\n".join(lines)

    def ask_question(self, question):
        """Ask a question about the resume and its analysis"""
        if not self.rag_vectorstore or not self.resume_text:
            return "Please analyze a resume first."

        retriever = self.rag_vectorstore.as_retriever(search_kwargs={"k": 4})
        qa_chain = self.build_retrieval_chain(
            retriever, extra_context=self._analysis_context()
        )

        response = qa_chain.invoke({"input": question})
        return response["answer"]

    @staticmethod
    def _strip_json_fence(raw, opener="{", closer="}"):
        """Pull a JSON blob out of a model reply.

        Models wrap JSON in ```json fences despite being told not to, which
        makes json.loads fail and pushes callers into lossy fallbacks.
        """
        text = (raw or "").strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) > 1:
                text = parts[1]
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        return text.strip()

    @staticmethod
    def _match_question_type(returned, question_types):
        """Map a model-returned type onto one of the requested types.

        Models paraphrase ("Technical Knowledge", "coding challenge"), so match
        case-insensitively in both directions. Returns None when nothing fits.
        """
        candidate = (returned or "").strip().lower()
        if not candidate:
            return None
        for requested in question_types:
            lowered = requested.lower()
            if lowered == candidate or lowered in candidate or candidate in lowered:
                return requested
        return None

    def generate_interview_questions_detailed(
        self,
        question_types,
        difficulty,
        num_questions,
        focus_skills=None,
        topics=None,
        seniority="",
        seniority_guidance="",
        target_role="",
    ):
        """Interview questions with an expected answer and a hint for each.

        Returns dicts of: type, topic, skill, question, expected_answer, hint.

        `topics` sets the subject areas to cover (system design, the person's
        own resume, challenges they hit, and so on) and `seniority` sets how
        deep each one goes — a system design question for an SDE2 is a
        different question from the same topic for an SDE1.
        """
        if not self.resume_text or not self.extracted_skills:
            return []

        skills = [s for s in (focus_skills or []) if s] or self.extracted_skills
        topic_list = [t for t in (topics or []) if t]

        try:
            llm = ChatOpenAI(model=self.model, api_key=self.api_key)

            context = f"""
                Resume Content:
                {self.resume_text[:2000]}...

                Skills to focus on: {', '.join(skills)}

                Strengths: {', '.join(self.analysis_result.get('strengths', []))}

                Areas for improvement: {', '.join(self.analysis_result.get('missing_skills', []))}
                """

            if topic_list:
                topic_block = (
                    f"Cover these topics, spreading the questions across them as evenly "
                    f"as the count allows:\n- " + "\n- ".join(topic_list)
                )
            else:
                topic_block = ""

            level_block = ""
            if seniority:
                level_block = f"Target level: {seniority}."
                if seniority_guidance:
                    level_block += f" {seniority_guidance}"
                level_block += (
                    " Pitch the depth, scope and expected answer to that level — the "
                    "same topic should be asked differently at a different level."
                )

            role_block = f"Target role: {target_role}." if target_role else ""

            # Spread the set across the requested types instead of letting the
            # model settle on whichever one it finds easiest.
            if len(question_types) > 1:
                spread = (
                    f"Distribute the {num_questions} questions across these types as evenly "
                    f"as possible, and use every one of them at least once if "
                    f"{num_questions} >= {len(question_types)}."
                )
            else:
                spread = f"Every question must be of type \"{question_types[0]}\"."

            prompt = f"""
                Generate exactly {num_questions} personalized {difficulty.lower()} level
                interview questions for this candidate. Use only these question types:
                {', '.join(question_types)}.

                {role_block}
                {level_block}

                {spread}

                {topic_block}

                {context}

                Ground the set in this specific candidate:
                - Ask at least one question about their actual background and career path
                  ("Resume Deep-Dive" / "Career Motivation" topics), naming a real company,
                  project or technology from the resume above.
                - Ask at least one about a concrete challenge, failure or trade-off they
                  faced, and what they would do differently.
                - Never invent experience the resume does not mention.

                Return ONLY a JSON array, no prose and no markdown fences. Each element:
                {{
                  "type": "one of: {', '.join(question_types)}",
                  "topic": "the single topic from the list above this question covers",
                  "skill": "the single skill from the focus list this question targets, or \"\" if none",
                  "question": "the full question text",
                  "expected_answer": "a strong answer in 2-4 sentences, pitched at the target level",
                  "hint": "one sentence nudging the candidate in the right direction"
                }}

                For coding questions include a clear problem statement in the question text.
                For system design questions state the scale and constraints to design for.
                """

            # A JSON array this time, so the fence stripper looks for brackets.
            raw = self._strip_json_fence(llm.invoke(prompt).content, "[", "]")
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                return []

            questions = []
            unmatched = 0
            for item in parsed:
                if not isinstance(item, dict) or not item.get("question"):
                    continue

                # Never report a type that wasn't asked for; fall back to the
                # first requested type so the label stays truthful to the request.
                matched = self._match_question_type(item.get("type"), question_types)
                if matched is None:
                    unmatched += 1
                    matched = question_types[0]

                # Same treatment as the type: never report a topic nobody asked for.
                topic = ""
                if topic_list:
                    topic = self._match_question_type(item.get("topic"), topic_list) or ""
                elif item.get("topic"):
                    topic = str(item["topic"])

                questions.append(
                    {
                        "type": matched,
                        "topic": topic,
                        "skill": str(item.get("skill") or ""),
                        "question": str(item["question"]),
                        "expected_answer": str(item.get("expected_answer") or ""),
                        "hint": str(item.get("hint") or ""),
                    }
                )

            if unmatched:
                print(
                    f"{unmatched} question(s) came back with a type outside "
                    f"{question_types}; relabelled as {question_types[0]}."
                )
            return questions

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Could not parse detailed interview questions: {e}")
            # Fall back to the older tuple-based generator so the user still
            # gets questions, just without answers or hints.
            return [
                {
                    "type": t,
                    "topic": "",
                    "skill": "",
                    "question": q,
                    "expected_answer": "",
                    "hint": "",
                }
                for t, q in self.generate_interview_questions(
                    question_types, difficulty, num_questions
                )
            ]
        except Exception as e:
            print(f"Error generating detailed interview questions: {e}")
            return []

    def generate_interview_questions(self, question_types, difficulty, num_questions):
        """Generate interview questions based on the resume"""
        if not self.resume_text or not self.extracted_skills:
            return []

        try:
            llm = ChatOpenAI(model=self.model, api_key=self.api_key)

            context = f"""
                Resume Content:
                {self.resume_text[:2000]}...

                Skills to focus on: {', '.join(self.extracted_skills)}

                Strengths: {', '.join(self.analysis_result.get('strengths', []))}

                Areas for improvement: {', '.join(self.analysis_result.get('missing_skills', []))}
                """

            prompt = f"""
                Generate {num_questions} personalized {difficulty.lower()} level interview questions for this candidate based on their resume and skills. Include only the following question types: {', '.join(question_types)}.

                For each question:
                1. Clearly label the question type
                2. Make the question specific to their background and skills
                3. For coding questions, include a clear problem statement

                {context}

                Format the response as a list of tuples with the question type and the question itself.
                Each tuple should be in the format: ("Question Type", "Full Question Text")
                """

            response = llm.invoke(prompt)
            questions_text = response.content

            questions = []
            pattern = r'\(["\']([^"\']+)["\'],\s+["\']([^"\']+)["\']\)\s+'
            matches = re.findall(pattern, questions_text, re.DOTALL)

            for match in matches:
                if len(match) >= 2:
                    question_type = match[0].strip()
                    question = match[1].strip()

                    for requested_type in question_types:
                        if requested_type.lower() in question_type.lower():
                            questions.append((requested_type, question))
                            break

            if not questions:
                lines = questions_text.split("\n")
                current_type = None
                current_question = ""

                for line in lines:
                    line = line.strip()
                    if (
                        any(t.lower() in line.lower() for t in question_types)
                        and not current_question
                    ):
                        current_type = next(
                            (t for t in question_types if t.lower() in line.lower()),
                            None,
                        )
                        if ":" in line:
                            current_question = line.split(":", 1)[1].strip()
                    elif current_type and line:
                        current_question += " " + line
                    elif current_type and current_question:
                        questions.append((current_type, current_question))
                        current_type = None
                        current_question = ""

                questions = questions[:num_questions]

            return questions

        except Exception as e:
            print(f"Error generating interview questions: {e}")
            return []

    def improve_resume(self, improvement_areas, target_role=""):
        """Generate suggestions to improve the resume"""
        if not self.resume_text:
            return {}

        try:
            improvements = {}

            for area in improvement_areas:

                if area == "Skills Highlighting" and self.resume_weaknesses:
                    skill_improvements = {
                        "description": "Your resume needs to better highlight key skills that are important for the role.",
                        "specific": [],
                    }

                    before_after_examples = {}

                    for weakness in self.resume_weaknesses:
                        skill_name = weakness.get("skill", "")
                        if "suggestions" in weakness and weakness["suggestions"]:
                            for suggestion in weakness["suggestions"]:
                                skill_improvements["specific"].append(
                                    f"**{skill_name}**: {suggestion}"
                                )

                        if "example" in weakness and weakness["example"]:

                            resume_chunks = self.resume_text.split("\n\n")
                            relevant_chunk = ""

                            for chunk in resume_chunks:
                                if (
                                    skill_name.lower() in chunk.lower()
                                    or "experience" in chunk.lower()
                                ):
                                    relevant_chunk = chunk
                                    break

                            if relevant_chunk:
                                before_after_examples = {
                                    "before": relevant_chunk.strip(),
                                    "after": relevant_chunk.strip()
                                    + "\n• "
                                    + weakness["example"],
                                }

                    if before_after_examples:
                        skill_improvements["before_after"] = before_after_examples

                    improvements["Skills Highlighting"] = skill_improvements

            remaining_areas = [
                area for area in improvement_areas if area not in improvements
            ]

            if remaining_areas:
                llm = ChatOpenAI(model=self.model, api_key=self.api_key)

                # Create a context with resume analysis and weaknesses
                weaknesses_text = ""
                if self.resume_weaknesses:
                    weaknesses_text = "Resume Weaknesses:\n"
                    for i, weakness in enumerate(self.resume_weaknesses):
                        weaknesses_text += (
                            f"{i+1}. {weakness['skill']}: {weakness['detail']}\n"
                        )
                        if "suggestions" in weakness:
                            for j, sugg in enumerate(weakness["suggestions"]):
                                weaknesses_text += f"   - {sugg}\n"

                context = f"""
                    Resume Content:
                    {self.resume_text}

                    Skills to focus on: {', '.join(self.extracted_skills)}

                    Strengths: {', '.join(self.analysis_result.get('strengths', []))}

                    Areas for improvement: {', '.join(self.analysis_result.get('missing_skills', []))}

                    {weaknesses_text}

                    Target role: {target_role if target_role else "Not specified"}
                    """

                prompt = f"""
                    Provide detailed suggestions to improve this resume in the following areas: {', '.join(remaining_areas)}.

                    {context}

                    For each improvement area, provide:
                    1. A general description of what needs improvement
                    2. 3-5 specific actionable suggestions
                    3. Where relevant, provide a before/after example

                    Format the response as a JSON object with improvement area keys, each containing:
                    - "description": general description
                    - "specific": list of specific suggestions
                    - "before_after": (where applicable) a dict with "before" and "after" examples

                    Only include the requested improvement areas that aren't already covered.
                    Focus particularly on addressing the resume weaknesses identified.
                    """

                response = llm.invoke(prompt)

                # Try to parse JSON from the response
                ai_improvements = {}
                try:
                    ai_improvements = json.loads(
                        self._strip_json_fence(response.content)
                    )
                    if not isinstance(ai_improvements, dict):
                        ai_improvements = {}

                    # Only keep areas that were actually requested, so a stray
                    # fence or brace can never surface as a heading in the UI.
                    for key, value in ai_improvements.items():
                        area = self._match_question_type(key, remaining_areas)
                        if area and isinstance(value, dict):
                            improvements[area] = value
                except (json.JSONDecodeError, AttributeError, TypeError):
                    ai_improvements = {}

                # If JSON parsing failed, create structured output manually
                if not ai_improvements:
                    sections = response.content.split("##")

                    for section in sections:
                        if not section.strip():
                            continue

                        lines = section.strip().split("\n")
                        area = None

                        for line in lines:
                            stripped = line.strip()
                            # Skip fences and bare punctuation from a failed JSON reply.
                            if stripped.startswith("```") or stripped in ("{", "}", "[", "]", ""):
                                continue
                            if not area:
                                matched = self._match_question_type(
                                    stripped.strip("#*: "), remaining_areas
                                )
                                if not matched:
                                    continue
                                area = matched
                                improvements.setdefault(
                                    area, {"description": "", "specific": []}
                                )
                            elif area and "specific" in improvements[area]:
                                if line.strip().startswith("- "):
                                    improvements[area]["specific"].append(
                                        line.strip()[2:]
                                    )
                                elif not improvements[area]["description"]:
                                    improvements[area]["description"] += line.strip()

            # Ensure all requested areas are included
            for area in improvement_areas:
                if area not in improvements:
                    improvements[area] = {
                        "description": f"Improvements needed in {area}",
                        "specific": ["Review and enhance this section"],
                    }

            return improvements

        except Exception as e:
            print(f"Error generating resume improvements: {e}")
            return {
                area: {"description": "Error generating suggestions", "specific": []}
                for area in improvement_areas
            }

    def get_improved_resume(self, target_role="", highlight_skills=""):
        """Generate an improved version of the resume optimized for the job description"""
        if not self.resume_text:
            return "Please upload and analyze a resume first."

        try:
            # Parse highlight skills if provided
            skills_to_highlight = []
            if highlight_skills:

                if len(highlight_skills) > 100:
                    self.jd_text = highlight_skills
                    try:
                        parsed_skills = self.extract_skills_from_jd(highlight_skills)
                        if parsed_skills:
                            skills_to_highlight = parsed_skills
                        else:
                            skills_to_highlight = [
                                s.strip()
                                for s in highlight_skills.split(",")
                                if s.strip()
                            ]
                    except:
                        skills_to_highlight = [
                            s.strip() for s in highlight_skills.split(",") if s.strip()
                        ]
                else:
                    skills_to_highlight = [
                        s.strip() for s in highlight_skills.split(",") if s.strip()
                    ]

            if not skills_to_highlight and self.analysis_result:

                skills_to_highlight = self.analysis_result.get("missing_skills", [])

                skills_to_highlight.extend(
                    [
                        skill
                        for skill in self.analysis_result.get("strengths", [])
                        if skill not in skills_to_highlight
                    ]
                )

            if self.extracted_skills:
                skills_to_highlight.extend(
                    [
                        skill
                        for skill in self.extracted_skills
                        if skill not in skills_to_highlight
                    ]
                )

            weakness_context = ""
            improvement_examples = ""

            if self.resume_weaknesses:
                weakness_context = "Address these specific weaknesses:\n"

                for weakness in self.resume_weaknesses:
                    skill_name = weakness.get("skill", "")
                    weakness_context += (
                        f"- {skill_name}: {weakness.get('detail', '')}\n"
                    )

                    if "suggestions" in weakness and weakness["suggestions"]:
                        weakness_context += "  Suggested improvements:\n"
                        for suggestion in weakness["suggestions"]:
                            weakness_context += f"  * {suggestion}\n"

                    if "example" in weakness and weakness["example"]:
                        improvement_examples += (
                            f"For {skill_name}: {weakness['example']}\n\n"
                        )

            llm = ChatOpenAI(model=self.model, temperature=0.7, api_key=self.api_key)

            jd_context = ""
            if self.jd_text:
                jd_context = f"Job Description:\n{self.jd_text}\n\n"
            elif target_role:
                jd_context = f"Target Role: {target_role}\n\n"

            prompt = f"""
                Rewrite and improve this resume to make it highly optimized for the target job.

                {jd_context}Original Resume:
                {self.resume_text}

                Skills to highlight (in order of priority): {', '.join(skills_to_highlight)}

                {weakness_context}

                Here are specific examples of content to add:
                {improvement_examples}

                Please improve the resume by:
                1. Adding strong, quantifiable achievements
                2. Highlighting the specified skills strategically for ATS scanning
                3. Addressing all the weakness areas identified with the specific suggestions provided
                4. Incorporating the example improvements provided above
                5. Structuring information in a clear, professional format
                6. Using industry-standard terminology
                7. Ensuring all relevant experience is properly emphasized
                8. Adding measurable outcomes and achievements

                Return only the improved resume text, with no commentary before or
                after it. Keep it tight enough to set on a single page: prefer
                fewer, stronger bullets over more of them.

                Format it as plain text with light markdown, in exactly this shape:

                Full Name
                Role | Role | Key skills
                City, Country | email | phone | linkedin | github

                ## PROFESSIONAL SUMMARY
                Three or four lines of prose, no bullets.

                ## PROFESSIONAL EXPERIENCE
                **Job Title - Company (Mon YYYY - Mon YYYY)**
                *City, Country*
                - Achievement, with a number in it wherever one is honest.
                - **Technologies:** comma-separated list

                ## TECHNICAL SKILLS
                **Category:** comma-separated list, one category per line

                ## CERTIFICATIONS
                - One per line

                ## EDUCATION
                **Degree, Field (YYYY - YYYY)**
                *Institution*

                The first three lines are the header and carry no markdown. Every
                section heading is a `##` line in upper case. Bullets start with
                `- `. No tables, no columns, no horizontal rules, no page numbers.
                """

            response = llm.invoke(prompt)
            improved_resume = response.content.strip()

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".txt", mode="w", encoding="utf-8"
            ) as tmp:
                tmp.write(improved_resume)
                self.improved_resume_path = tmp.name

            return improved_resume

        except Exception as e:
            print(f"Error generating improved resume: {e}")
            return "Error generating improved resume. Please try again."

    def cleanup(self):
        """Clean up temporary files"""
        try:
            if hasattr(self, "resume_file_path") and os.path.exists(
                self.resume_file_path
            ):
                os.unlink(self.resume_file_path)

            if hasattr(self, "improved_resume_path") and os.path.exists(
                self.improved_resume_path
            ):
                os.unlink(self.improved_resume_path)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")
