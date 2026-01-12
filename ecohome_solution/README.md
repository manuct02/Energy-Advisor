# Energy Advisor

## Resumen del proyecto

Estamos construyendo la siguiente generación de sistemas de optimización de energía AI-powered que ayude a los clientes con paneles solares, coches eléctricos y smarth home dispositivos a reducir el coste eléctrico e imapcto global.

Construiremos un tutor de energía agéntico capaz de analizar datos en tiempo real, entender patrones de energía, y proveer recomendaciones personalizadas para un uso más eficiente de los recursos energéticos.

Necesitaremos un sistema capaz de:
- Analizar los patrones de uso de energía y los datos de generación solar.
- Aportar recomendaciones personalizadas para cuándo usar según qué dispositivos.
- Recomedar estrategias de ahorro de energía basadas en predicciones meteorológicas y precios de electricidad.
- Ayudar a los clientes a maximizar su uso de la energía solar.
- Calcular potenciales ahorros desde la optimización.

No solo debe aportar consejos, también debería tomar decisiones basadas en datos sobre la optimización de la energía.

### **Introducción**

**En este proyecto se desarrollará el *EcoHome Energy Advisor*, un agente inteligente capaz de optimizar el uso de la energía a entre varios dispositivos de smart home y sistemas**

### **Capacidades clave**
- Integración Multi-Tool: predicciones meteorológicas, costeo de la electricidad, queries sobre el uso de la energía y datos de la generación solar.
- RAG Pipeline: un RAG para consejos de ahorro de energía y mejores prácticas.
- Análisis histórico: tiene que tener en cuenta los patrones de energía previos para recomendaciones personalizadas.
- Optimización de coste: considerar el precio de la electricidad al día y la generación solar para recomendaciones.
- Guardar los cálculos.

### **Inputs**
- Datos sobre el uso de la energía (consumición, tipo de dispositivos, costes)
- Datos sobre la generación solar (producción, condiciones meteorológicas)
- Predicciones del tiempo y precios eléctricos.
- Conocimiento básico sobre el ahorro de la energía.
- Preguntas sobre la optimización de la energía.

### **Entregables**
Un sistema agéntico impulsado por LangGraph que:
- Entienda las preguntas sobre optimización de la energía.
- Recupere datos relevantes usando las herramientas apropiadas.
- Aporte recomendaciones personalizadas basadas en análisis de datos.
- Cite consejos sobre el ahorro de energía que saque de su conocimiento base.
- Calcule potenciales ahorros e impacto medioambiental.

## Instrucciones del Proyecto

### Project Structure

```
ecohome_starter/
├── models/
│   ├── __init__.py
│   └── energy.py              # Database models for energy data
├── data/
│   └── documents/
│       ├── tip_device_best_practices.txt
│       └── tip_energy_savings.txt
├── agent.py                   # Main Energy Advisor agent
├── tools.py                   # Agent tools (weather, pricing, database, RAG)
├── requirements.txt           # Python dependencies
├── 01_db_setup.ipynb         # Database setup and sample data
├── 02_rag_setup.ipynb        # RAG pipeline setup
├── 03_agent_evaluation.ipynb # Agent testing and evaluation
├── 04_agent_run.ipynb        # Running the agent with examples
└── README.md                  # This file
```

### **Setup**

- 1 - Runear el notebook `01_db_setuo.ipynb` para inicializar la base de datos y rellenarla con una muestra del uso de energía y datos de generación solar.
- 2 - Runear el notebook `02_rag_setup.ipynb` para setear el pipeline del RAG con consejos de ahorro de energía y buenas prácticas.

- 3 - Expandir el conocimiento base añadiendo **por lo menos 5 documentos de ahorro de energía adicionales** a la carpeta `data/documents`. Asegurar que cubramos los siguientes topics:
  - Estrategias de optimización HVAC (sea lo que sea eso)
  - Consejos de automatización de Smart House.
  - Integración de energías renovables.
  - Manejo de la energía estacional.
  - Optimización del almacenamiento de la energía.

### **Desarrollo del agente**

1. Echarle un ojo a las herramientas en `tools.py` para entender las capacidades disponibles.
2. Mejorar el agente en `agent.py` haciendo:
- Crear un sistema de instrucciones comprensible para el Energy Advisor.
- Implementar un manejo de errores apropiado.
- Añadir contexto de conciencia para mejores recomendaciones.

3. Evaluar y testear el agente usando los escenarios en `03_run_and_evaluate.ipynb`

### **Features clave para implementar**
- Integración del tiempo: usar predicciones meteorológicas para predecir la generación solar y optimizar el horario de los dispositivos.
- Cotización dinámica: tiene que considerar los precios del día a la hora de calcular ahorros eléctricos y optimización de costes.
- Análisis histórico: query de los usos previos de la energía.
- RAG Pipeline: Retrieve relevant energy-saving tips and best practices.
- Optimización multi-device: tiene que manejar EVs, HVAC, sistemas solares y aplicaciones.
- Cálculo de costes: Proporciona ahorros específicos estimados y análisis de ROI.


![alt text](image.png)

