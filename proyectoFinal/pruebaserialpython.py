"""
import serial
import time

# Configura el puerto serial
ser = serial.Serial('COM9', 9600)  

# Solicitar al usuario que ingrese un carácter
result = input("Por favor, ingresa (¨A¨ o ¨B¨): ")

if result == 'A':
    ser.write(b'A')  # Enviar 'A' si el resultado es 'A'
else:
    ser.write(b'B')  # Enviar 'B' si el resultado es 'B'

time.sleep(1)  # Espera un poco para enviar el dato
ser.close()  # Cierra el puerto serial
"""


import serial
import time

# Configura el puerto serial
ser = serial.Serial('COM9', 9600)

try:
    while True:  # Bucle infinito para solicitar constantemente la entrada del usuario
        # Solicitar al usuario que ingrese un carácter
        result = input("Por favor, ingresa (A, B, C, D, o F): ")

        # Verificar que el carácter sea uno de los esperados antes de enviar
        if result in ['A', 'B', 'C', 'D', 'F']:
            ser.write(result.encode())  # Enviar el carácter
            print(f"Enviado: {result}")
        else:
            print("Carácter no válido. Por favor, ingresa A, B, C, D, o F.")
        
        time.sleep(0.5)  # Pequeña pausa para estabilidad

except KeyboardInterrupt:
    print("Programa terminado por el usuario.")

finally:
    ser.close()  # Asegúrate de cerrar el puerto serial
    print("Puerto serial cerrado.")