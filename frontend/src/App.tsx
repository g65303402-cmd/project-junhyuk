import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { DevPage } from './pages/DevPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { UserCounselPage } from './pages/UserCounselPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<UserCounselPage />} />
        <Route path="/dev" element={<DevPage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
