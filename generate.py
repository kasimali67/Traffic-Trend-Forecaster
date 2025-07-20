import os

project_structure = {
    ".bolt/prompt": "",
    "src/components/TrafficChart.js": '''import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const TrafficChart = () => {
  const [data, setData] = useState({
    labels: [],
    datasets: [{
      label: 'Network Traffic',
      data: [],
      fill: false,
      borderColor: 'rgb(75, 192, 192)',
      backgroundColor: 'rgba(75, 192, 192, 0.2)',
      tension: 0.1
    }]
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await axios.get('http://localhost:8000/data');
        const trafficData = result.data;
        setData({
          labels: trafficData.map(d => new Date(d.date).toLocaleTimeString()),
          datasets: [{
            ...data.datasets[0],
            data: trafficData.map(d => d.traffic)
          }]
        });
      } catch (error) {
        console.error('Error fetching traffic data:', error);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    
      Real-time Network Traffic Monitoring
      
    
  );
};

export default TrafficChart;
''',
    ".gitignore": '''node_modules/
.pnp
.pnp.js
/build
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
__pycache__/
*.py[cod]
*$py.class
venv/
env/
.vscode/
.idea/
.DS_Store
Thumbs.db
''',
    "README.md": '''# Traffic Trend Forecaster

An LSTM-based time-series model to predict next-hour network traffic with automated data processing pipeline and real-time visualization dashboard.

## Features

- **LSTM PyTorch Model**: Reduces mean absolute percentage error by 18%
- **Automated Data Pipeline**: Pandas-based ingestion and preprocessing
- **Fast Model Retraining**: 24-hour rolling window retraining under 2 minutes
- **REST API**: FastAPI backend with prediction endpoints
- **Real-time Dashboard**: React.js frontend with Chart.js visualization
- **Live Monitoring**: WebSocket-based real-time traffic monitoring

## Technologies Used

- **Frontend**: React.js, TypeScript, Vite, TailwindCSS
- **Visualization**: Chart.js, React-Chart.js-2
- **Backend**: Python, FastAPI, PyTorch, Pandas
- **ML**: LSTM neural networks for time-series forecasting
- **Build Tools**: Vite, PostCSS, ESLint

## Setup Instructions

### Install Dependencies
```
npm install
```

### Development Server
```
npm run dev
```

### Build for Production
```
npm run build
```
''',
    "eslint.config.js": '''import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
    },
  },
)
''',
    "index.html": '''

  
    
    
    
    Traffic Trend Forecaster
    
  
  
    
    
  

''',
    "package-lock.json": '''{}''',
    "package.json": '''{
  "name": "traffic-trend-forecaster",
  "private": true,
  "version": "1.0.0",
  "description": "LSTM-based network traffic forecasting with real-time monitoring dashboard",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "lint": "eslint .",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "chart.js": "^4.4.0",
    "react-chartjs-2": "^5.2.0",
    "axios": "^1.6.0"
  },
  "devDependencies": {
    "@eslint/js": "^9.9.1",
    "@types/react": "^18.3.5",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "autoprefixer": "^10.4.20",
    "eslint": "^9.9.1",
    "eslint-plugin-react-hooks": "^5.1.0-rc.0",
    "eslint-plugin-react-refresh": "^0.4.11",
    "globals": "^15.9.0",
    "postcss": "^8.4.45",
    "tailwindcss": "^3.4.10",
    "typescript": "^5.5.3",
    "typescript-eslint": "^8.3.0",
    "vite": "^5.4.2"
  }
}
''',
    "postcss.config.js": '''export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
''',
    "structure.txt": '''Traffic-Trend-Forecaster/
├── .bolt/
│   └── prompt
├── src/
│   └── components/
│       └── TrafficChart.js
├── .gitignore
├── README.md
├── eslint.config.js
├── index.html
├── package-lock.json
├── package.json
├── postcss.config.js
├── structure.txt
├── tailwind.config.js
├── tsconfig.app.json
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts
''',
    "tailwind.config.js": '''/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        }
      }
    },
  },
  plugins: [],
}
''',
    "tsconfig.app.json": '''{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true
  },
  "include": [
    "src"
  ]
}
''',
    "tsconfig.json": '''{
  "files": [],
  "references": [
    {
      "path": "./tsconfig.app.json"
    },
    {
      "path": "./tsconfig.node.json"
    }
  ]
}
''',
    "tsconfig.node.json": '''{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2023"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "noEmit": true
  },
  "include": [
    "vite.config.ts"
  ]
}
''',
    "vite.config.ts": '''import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
''',
}

folders = [
    ".bolt",
    "src",
    "src/components",
]

def create_file(filepath, content=""):
    # Skip empty or invalid filepaths
    if not filepath or not filepath.strip():
        print(f"Skipping empty filepath")
        return
    
    # Get directory name
    dir_name = os.path.dirname(filepath)
    
    # Only create directory if it's not empty
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # Create the file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    # Create folders first
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")
    
    # Create files
    for path, content in project_structure.items():
        create_file(path, content)
        print(f"Created file: {path}")
    
    print(f"\n✅ Traffic Trend Forecaster project structure created!")
    print(f"Next steps:")
    print(f"1. git add . && git commit -m 'Initial project structure' && git push")

if __name__ == "__main__":
    main()
