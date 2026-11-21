CREATE TABLE kunden (
  id INT PRIMARY KEY,
  name NVARCHAR(100) NOT NULL,
  ort NVARCHAR(60) NOT NULL
);
CREATE TABLE projekte (
  id NVARCHAR(10) PRIMARY KEY,            -- 'P-2041'
  name NVARCHAR(150) NOT NULL,
  kunde_id INT NOT NULL REFERENCES kunden(id),
  status NVARCHAR(30) NOT NULL,           -- Angebot | Planung | In Ausführung | Abgeschlossen
  projektleiter NVARCHAR(80) NOT NULL,
  start_datum DATE NOT NULL,
  ende_geplant DATE NOT NULL,
  budget_eur DECIMAL(12,2) NOT NULL
);
CREATE TABLE stunden (
  id INT PRIMARY KEY,
  projekt_id NVARCHAR(10) NOT NULL REFERENCES projekte(id),
  mitarbeiter NVARCHAR(80) NOT NULL,
  datum DATE NOT NULL,
  stunden DECIMAL(4,1) NOT NULL,
  taetigkeit NVARCHAR(60) NOT NULL
);
CREATE TABLE material (
  id INT PRIMARY KEY,
  projekt_id NVARCHAR(10) NOT NULL REFERENCES projekte(id),
  bezeichnung NVARCHAR(120) NOT NULL,
  menge DECIMAL(10,2) NOT NULL,
  einheit NVARCHAR(10) NOT NULL,
  kosten_eur DECIMAL(12,2) NOT NULL,
  lieferant NVARCHAR(80) NOT NULL
);
CREATE TABLE offene_punkte (
  id NVARCHAR(10) PRIMARY KEY,            -- 'OP-001'
  projekt_id NVARCHAR(10) NOT NULL REFERENCES projekte(id),
  beschreibung NVARCHAR(250) NOT NULL,
  prioritaet NVARCHAR(10) NOT NULL,       -- hoch | mittel | niedrig
  status NVARCHAR(20) NOT NULL,           -- offen | in Arbeit | erledigt
  faellig_am DATE NULL,
  verantwortlich NVARCHAR(80) NOT NULL
);