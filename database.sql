DROP TABLE IF EXISTS Record;

CREATE TABLE IF NOT EXISTS Record
(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL
);

INSERT INTO Record (Name) VALUES
    ('16265'), ('16272'), 
    ('16273'), ('16420'), 
    ('16483'), ('16539'), 
    ('16773'), ('16786'), 
    ('16795'), ('17052'), 
    ('17453'), ('18177'), 
    ('18184'), ('19088'), 
    ('19090'), ('19093'),
    ('19140'), ('19830');

CREATE TABLE IF NOT EXISTS ExperimentParameters
(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    SignalDuration INTEGER NOT NULL,
    Notes TEXT
);

INSERT INTO ExperimentParameters (SignalDuration, Notes) VALUES
    (640, 'Median filter used with kernel size of 9.');

CREATE TABLE IF NOT EXISTS KeyMetaData
(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    KeyInBits TEXT,
    KeyErrors TEXT,
    KeyBitFlips INTEGER
);

CREATE TABLE IF NOT EXISTS KeyAgreementInstance
(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    SampleFrom INTEGER NOT NULL,
    K1MetaDataID INTEGER NOT NULL,
    K2MetaDataID INTEGER NOT NULL,
    PredictionCoefficients TEXT,
    BCHCoefficients TEXT,
    TargetPeakLocations TEXT,
    FOREIGN KEY (K1MetaDataID)
        REFERENCES KeyMetaData (ID),
    FOREIGN KEY (K2MetaDataID)
        REFERENCES KeyMetaData (ID)
);

CREATE TABLE IF NOT EXISTS Experiment
(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    RecordID INTEGER NOT NULL,
    ExperimentParametersID INTEGER NOT NULL,
    KeyAgreementInstanceID INTEGER NOT NULL,
    FOREIGN KEY (RecordID)
        REFERENCES Record (ID),
    FOREIGN KEY (ExperimentParametersID)
        REFERENCES ExperimentParameters (ID),
    FOREIGN KEY (KeyAgreementInstanceID)
        REFERENCES KeyAgreementInstance (ID)
);

CREATE TABLE IF NOT EXISTS AttackParameters
(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    PeakDistanceLower INTEGER NOT NULL,
    PeakDistanceUpper INTEGER NOT NULL,
    PeakDistanceStride INTEGER NOT NULL,
    QRSLength INTEGER NOT NULL
);

INSERT INTO AttackParameters (PeakDistanceLower, PeakDistanceUpper, PeakDistanceStride, QRSLength) VALUES
    (60, 80, 1, 60);

CREATE TABLE IF NOT EXISTS Attack
(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    AttackParametersID INTEGER NOT NULL,
    ExperimentID INTEGER NOT NULL,
    LeadName TEXT NOT NULL,
    OffsetAsSeconds INTEGER NOT NULL,
    K3MetaDataID INTEGER NOT NULL,
    K4MetaDataID INTEGER NOT NULL,
    FOREIGN KEY (AttackParametersID)
        REFERENCES AttackParameters (ID),
    FOREIGN KEY (ExperimentID)
        REFERENCES Experiment (ID),
    FOREIGN KEY (K3MetaDataID)
        REFERENCES KeyMetaData (ID),
    FOREIGN KEY (K4MetaDataID)
        REFERENCES KeyMetaData (ID)
);

CREATE TABLE IF NOT EXISTS Attempt
(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    AttackID INTEGER NOT NULL,
    K5MetaDataID INTEGER NOT NULL,
    PeakPositions TEXT NOT NULL,
    FOREIGN KEY (AttackID)
        REFERENCES Attack (ID),
    FOREIGN KEY (K5MetaDataID)
        REFERENCES KeyMetaData (ID)
);