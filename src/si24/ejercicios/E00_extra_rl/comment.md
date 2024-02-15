
```
        self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + self.gamma*max(self.Q[next_state] -  self.Q[state][action]))

```
En esta linea de código, el máx es solo sobre el `self.Q[next_state]`:

```
        self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + self.gamma*max(self.Q[next_state]) -  self.Q[state][action])

```
