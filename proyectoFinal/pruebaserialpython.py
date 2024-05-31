import serial
import time

# Configura el puerto serial
ser = serial.Serial('COM11', 9600)

try:
    while True:  # Bucle infinito para solicitar constantemente la entrada del usuario
        # Solicitar al usuario que ingrese un carácter
        result = input("Por favor, ingresa (1, 2, 5, 11, ): ")

        # Verificar que el carácter sea uno de los esperados antes de enviar
        if result in ['0', '1', '2', '5', '11', '13', '6']:
            ser.write(result.encode())  # Enviar el carácter
            print(f"Enviado: {result}")
        else:
            print("Carácter no válido. Por favor, ingresa 0, 1, 2, 3, o 4.")
        
        time.sleep(0.5)  # Pequeña pausa para estabilidad

except KeyboardInterrupt:
    print("Programa terminado por el usuario.")

finally:
    ser.close()  # Asegúrate de cerrar el puerto serial
    print("Puerto serial cerrado.")