from pydantic import BaseModel
from typing import List, Optional, Union
from uuid import UUID, uuid4


class AudioUploader(BaseModel):
    # text_base64_content: Optional[Union[str, List[str]]] = None
    audio_base64_content: Optional[Union[str, List[str]]] = None
    langauge: str = None
    # audio_txt: str = None
    # image_base64_content: Optional[Union[str, List[str]]] = None
    
    
class RegisterationPage(BaseModel):
    email: str
    password: str
    disability: Optional[str] = None
    language: Optional[str] = None 
    status: str = None

class LoginPage(BaseModel):
    email: str
    password: str
    status: str = None