# Importaciones para ejecutar FastAPI y uvicorn
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os

# Importaciones para el uso del chat con langchain
from langchain_core.messages import HumanMessage
from agent_config import app_flujo, config, detectar_idioma, traducir_respuesta
from langchain.schema import OutputParserException

app = FastAPI(title='Asistente de viajes (Cambiando el Rumbo)', version='1.0.0')

# Montamos la carpeta static para que sea accesible en la ruta /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define la ruta absoluta de la carpeta 'templates'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint para el chatbot
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_input = request.message

    try:
        # detectamos el idioma de entrada del usuario
        idioma_entrada=detectar_idioma(user_input)

        # generamos la respuetsa con nuestro flujo que gestiona el agente y la memoria.
        response = app_flujo.invoke({"messages" : [HumanMessage(content=user_input)]}, config=config)

        # guardamos la respuesta de nuestro agente, en una variable para acceder a ella facilmente
        resultado = response["messages"][-1].content

        if idioma_entrada == 'es':

            return {"response": resultado}

        else:
            resultado_traducido = traducir_respuesta(resultado, idioma_entrada)

            return {"response": resultado_traducido}

    except OutputParserException:
        raise HTTPException(status_code=400, detail="No he podido procesar la consulta adecuadamente. ¿Podrías intentar una entrada diferente?")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Punto de entrada para arrancar el servidor con Python
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )