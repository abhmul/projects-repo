from dataclasses import dataclass
from string import punctuation
import re
from enum import Enum

class InputType(Enum):
    SEMESTER_COURSES = "semester"
    CATALOG_COURSES = "catalog"

    def __str__(self):
        return self.value

@dataclass
class Parseable:
    text: str

    @staticmethod
    def parse(text: str, input_type: InputType) -> 'Parseable':
        NotImplemented

@dataclass
class SectionsTable(Parseable):
    """Class for storing and parsing the table of sections for a course"""
    # TODO: Fill this out as needed

    @staticmethod
    def parse(text: str, input_type: InputType) -> 'SectionsTable':
        return SectionsTable(text)
    
    @staticmethod
    def empty() -> 'SectionsTable':
        return SectionsTable("")

@dataclass
class CourseDescription(Parseable):
    """Class for storing and parsing the description of a course"""

    @staticmethod
    def parse(text: str, input_type: InputType) -> 'CourseDescription':
        CourseDescription(text)
    
    @staticmethod
    def empty() -> 'CourseDescription':
        return CourseDescription("")

@dataclass
class Prerequisites(Parseable):
    """Prerequisites for a course"""
    codes: list
    # departmental_approval_required: bool
    # instructor_approval_required: bool

    @staticmethod
    def parse(text: str, input_type: InputType) -> 'Prerequisites':
        codes = re.findall(COURSE_CODE_RE, text)
        return Prerequisites(text, codes)

    @staticmethod
    def empty() -> 'Prerequisites':
        return Prerequisites("", [])

@dataclass
class Recommendations(Parseable):
    """Recommendation Background for a course"""
    codes: list

    @staticmethod
    def parse(text: str, input_type: InputType) -> 'Recommendations':
        codes = re.findall(COURSE_CODE_RE, text)
        return Recommendations(text, codes)

    @staticmethod
    def empty() -> 'Recommendations':
        return Recommendations("", [])

@dataclass
class Hours(Parseable):
    """Number of hours a course can be"""
    # hours: list

    @staticmethod
    def parse(text: str, input_type: InputType) -> 'Hours':
        return Hours(text)

    @staticmethod
    def empty() -> 'Hours':
        return Hours("")

@dataclass
class Code(Parseable):
    """Code of a course"""
    subject: str
    number: int

    @staticmethod
    def parse(text: str, input_type: InputType) -> 'Hours':
        subject, number = text.split()
        number = int(number)
        return Code(text=text, subject=subject, number=number)

    @staticmethod
    def empty() -> 'Hours':
        return Hours("")

@dataclass
class Course(Parseable):
    """Class for keeping track of an item in inventory."""
    code: Code
    title: str
    description: CourseDescription = CourseDescription.empty()
    hours: Hours = Hours.empty()
    prerequisites: Prerequisites = Prerequisites.empty()
    recommendations: Recommendations = Recommendations.empty()
    sections: SectionsTable = SectionsTable.empty()

    @staticmethod
    def parse(text: str, input_type: InputType) -> 'CourseDescription':
        if input_type == InputType.SEMESTER_COURSES:
            return Course.parse_semester_course(text)
        elif input_type == InputType.CATALOG_COURSES:
            return Course.parse_course_catalog(text)
        else:
            NotImplemented
    
    @staticmethod
    def parse_course_catalog(text: str) -> 'Course':
        lines = text.splitlines()
        print(lines[0])
        print(len(lines))
        header_line = lines[0].split('.')
        code = Code.parse(header_line[0].strip(), InputType.CATALOG_COURSES)
        title = header_line[1].strip()
        hours = Hours.parse(header_line[2], InputType.CATALOG_COURSES)
        details = lines[2].split("Prerequisite(s): ")
        course_description = CourseDescription.parse(details[0], InputType.CATALOG_COURSES)
        if len(details) > 1:
            prerequisites = Prerequisites.parse(details[1], InputType.CATALOG_COURSES)
        else:
            prerequisites = Prerequisites.empty()
        
        return Course(text=text, code=code, title=title, hours=hours, description=course_description, prerequisites=prerequisites)

        

    @staticmethod
    def parse_semester_course(text: str) -> 'Course':
        def parse_course_details(details: str) -> tuple:
            hours = Hours.parse(details[:details.find(" hours")], InputType.SEMESTER_COURSES)
            
            prereq_key = "Prerequisite(s): "
            prereq_loc = details.find(prereq_key)
            if prereq_loc != -1:
                prerequisites = details[prereq_loc + len(prereq_key):].strip(punctuation)
            else:
                prerequisites = ""
            prerequisites = Prerequisites.parse(prerequisites, InputType.SEMESTER_COURSES)

            background_key = "Recommended background: "
            background_loc = details.find(background_key)
            if background_loc != -1:
                recommended_background = details[background_loc + len(background_key):].strip(punctuation)
            else:
                recommended_background = ""
            recommendations = Recommendations.parse(recommended_background, InputType.SEMESTER_COURSES)

            return hours, prerequisites, recommendations

        lines = text.splitlines()
        # print(lines[:3])
        code = Code.parse(lines[0], InputType.SEMESTER_COURSES)
        # print(code)
        title = lines[1]
        hours, prerequisites, recommendations = parse_course_details(lines[2])
        sections = SectionsTable.parse(lines[4], InputType.SEMESTER_COURSES)

        return Course(text=text, code=code, title=title, hours=hours, prerequisites=prerequisites, recommendations=recommendations, sections=sections)
    
    def to_dict(self) -> dict:
        return dict(
            subject = self.code.subject,
            number = self.code.number,
            title = self.title,
            prerequisites = self.prerequisites.text,
            recommended = self.recommendations.text
        )

COURSE_TYPES = ["MATH", "MCS", "STAT", "PHYS"]
COURSE_CODE_RE = f"(?:{'|'.join(COURSE_TYPES)}) \d\d\d"

def parse_courses(raw_text: str, input_type: InputType):
    if input_type == InputType.SEMESTER_COURSES:
        return parse_semester_courses_list(raw_text)
    elif input_type == InputType.CATALOG_COURSES:
        return parse_catalog_courses_list(raw_text)

def parse_semester_courses_list(raw_text: str):
    courses = re.split(f"(?={COURSE_CODE_RE}\n)", raw_text)[1:]  # First element is always empty string
    courses = [Course.parse(c, InputType.SEMESTER_COURSES) for c in courses]
    return courses

def parse_catalog_courses_list(raw_text: str):
    search_str = f"(?=^{COURSE_CODE_RE}\. )"
    courses = re.split(re.compile(search_str, re.MULTILINE), raw_text)[1:]
    courses = [Course.parse(c, InputType.CATALOG_COURSES) for c in courses]
    return courses