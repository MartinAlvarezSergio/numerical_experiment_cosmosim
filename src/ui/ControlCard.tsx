import { ReactNode } from "react";

type ControlCardProps = {
  title: string;
  subtitle?: string;
  children: ReactNode;
};

export function ControlCard({ title, subtitle, children }: ControlCardProps): JSX.Element {
  return (
    <section className="panel card">
      <h3>{title}</h3>
      {subtitle ? <p className="subtle">{subtitle}</p> : null}
      {children}
    </section>
  );
}
