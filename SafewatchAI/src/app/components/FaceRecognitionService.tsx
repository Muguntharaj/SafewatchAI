// Face Recognition Service - Simulates face matching using simplified face signatures

export interface FaceSignature {
  personId: string;
  personName: string;
  category: 'Employee' | 'Owner' | 'Unknown';
  signature: number[]; // Simulated face embedding (in real app, would be neural network output)
  timestamp: Date;
}

export class FaceRecognitionService {
  private faceDatabase: FaceSignature[] = [];
  private readonly MATCH_THRESHOLD = 0.85; // Similarity threshold for face matching

  // Generate a simulated face signature from detection
  generateFaceSignature(detection: any): number[] {
    // In a real app, this would use a face recognition model like FaceNet or ArcFace
    // For demo, we create a random but consistent signature based on detection properties
    const seed = detection.id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const signature: number[] = [];
    
    for (let i = 0; i < 128; i++) {
      // Generate pseudo-random but deterministic values
      signature.push(Math.sin(seed * (i + 1)) * Math.cos(seed / (i + 1)));
    }
    
    return signature;
  }

  // Calculate similarity between two face signatures (cosine similarity)
  calculateSimilarity(sig1: number[], sig2: number[]): number {
    if (sig1.length !== sig2.length) return 0;

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < sig1.length; i++) {
      dotProduct += sig1[i] * sig2[i];
      norm1 += sig1[i] * sig1[i];
      norm2 += sig2[i] * sig2[i];
    }

    const similarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    return similarity;
  }

  // Try to match a face signature against the database
  matchFace(faceSignature: number[]): FaceSignature | null {
    let bestMatch: FaceSignature | null = null;
    let bestSimilarity = 0;

    for (const knownFace of this.faceDatabase) {
      const similarity = this.calculateSimilarity(faceSignature, knownFace.signature);
      
      if (similarity > bestSimilarity && similarity >= this.MATCH_THRESHOLD) {
        bestSimilarity = similarity;
        bestMatch = knownFace;
      }
    }

    console.log('Face matching result:', {
      found: bestMatch !== null,
      similarity: bestSimilarity,
      match: bestMatch?.personName
    });

    return bestMatch;
  }

  // Register a new face in the database
  registerFace(personId: string, personName: string, category: 'Employee' | 'Owner', faceSignature: number[]): void {
    // Check if person already exists
    const existingIndex = this.faceDatabase.findIndex(f => f.personId === personId);
    
    const newFace: FaceSignature = {
      personId,
      personName,
      category,
      signature: faceSignature,
      timestamp: new Date()
    };

    if (existingIndex >= 0) {
      // Update existing entry
      this.faceDatabase[existingIndex] = newFace;
    } else {
      // Add new entry
      this.faceDatabase.push(newFace);
    }

    console.log('Face registered:', personName, 'Total faces in DB:', this.faceDatabase.length);
  }

  // Update a person's classification
  updatePersonCategory(personId: string, personName: string, category: 'Employee' | 'Owner'): void {
    const face = this.faceDatabase.find(f => f.personId === personId);
    if (face) {
      face.personName = personName;
      face.category = category;
    }
  }

  // Get all registered faces
  getAllFaces(): FaceSignature[] {
    return [...this.faceDatabase];
  }

  // Clear the database (for testing)
  clearDatabase(): void {
    this.faceDatabase = [];
  }
}

// Singleton instance
export const faceRecognitionService = new FaceRecognitionService();
