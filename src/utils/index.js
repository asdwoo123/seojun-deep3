import { ipcRenderer } from 'electron'

function mainConsole(message) {
    ipcRenderer.send('console', message)
}

export {
    mainConsole
}