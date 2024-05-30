// Define los pines de los LEDs
int ledPinA = 12;  
int ledPinB = 11;  
int ledPinC = 10;  
int ledPinD = 9; 

void setup() {
  pinMode(ledPinA, OUTPUT);
  pinMode(ledPinB, OUTPUT);
  pinMode(ledPinC, OUTPUT);
  pinMode(ledPinD, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(9600);  // Inicia la comunicación serial a 9600 baudios
}

void loop() {
  if (Serial.available() > 0) {
    char received = Serial.read();  // Lee el carácter recibido
  
    if (received == '0') {
      digitalWrite(ledPinA, HIGH);
      digitalWrite(ledPinB, LOW);
      digitalWrite(ledPinC, LOW);
      digitalWrite(ledPinD, LOW);
    } 
    
    if (received == '1') {
      digitalWrite(ledPinA, LOW);
      digitalWrite(ledPinB, HIGH);
      digitalWrite(ledPinC, LOW);
      digitalWrite(ledPinD, LOW);
    }

    if (received == '2') {
      digitalWrite(ledPinA, LOW);
      digitalWrite(ledPinB, LOW);
      digitalWrite(ledPinC, HIGH);
      digitalWrite(ledPinD, LOW);
    }

    if (received == '3') {
      digitalWrite(ledPinA, LOW);
      digitalWrite(ledPinB, LOW);
      digitalWrite(ledPinC, LOW);
      digitalWrite(ledPinD, HIGH);
    }

    if (received == '4') {
      digitalWrite(ledPinA, LOW);
      digitalWrite(ledPinB, LOW);
      digitalWrite(ledPinC, LOW);
      digitalWrite(ledPinD, LOW);
    }

    }

  digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(500);                       // wait for a second
  digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
  delay(500);   
  
}
