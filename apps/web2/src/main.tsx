import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
<<<<<<< HEAD
import { BrowserRouter } from 'react-router-dom'
=======
>>>>>>> a8121246fd2bb72b9a821923dc0a2bf48f07239c
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
<<<<<<< HEAD
    <BrowserRouter>
      <App />
    </BrowserRouter>
=======
    <App />
>>>>>>> a8121246fd2bb72b9a821923dc0a2bf48f07239c
  </StrictMode>,
)
