import { UniverseICsCanvas } from "../applets/cosmic_ics/UniverseICsCanvas";

export function App(): JSX.Element {
  return (
    <div className="app-shell">
      <main>
        <section className="modal card">
          <UniverseICsCanvas />
        </section>
      </main>
    </div>
  );
}
