import { ModelPanels, ModelExplanation } from '../components/model/ModelPanels';
import { useApp } from '../store/AppContext';

export function ModelsPage() {
  const { twinEngineId } = useApp();
  return (
    <div className="grid gap-3 pb-24 lg:grid-cols-12">
      <div className="lg:col-span-8"><ModelPanels /></div>
      <div className="lg:col-span-4"><ModelExplanation engineId={twinEngineId} /></div>
    </div>
  );
}
