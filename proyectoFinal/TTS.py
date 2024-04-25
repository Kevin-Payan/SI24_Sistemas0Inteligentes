import elevenlabs

#Busca la manera apropiada de esconder esta key
elevenlabs.set_api_key("")

#Generate Audio 

audio = elevenlabs.generate(
    text = "¡Hola, buenos dias gente!...¿Cómo se encuentran el dia de hoy?",
    voice = "Chris",
    model = "eleven_multilingual_v2"
)

#Play Audio

elevenlabs.play(audio)






