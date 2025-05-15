import SwiftUI

struct MainView: View {
    var body: some View {
        NavigationStack {
            ZStack {
                LinearGradient(gradient: Gradient(colors: [.blue.opacity(0.2), .gray.opacity(0.1)]),
                               startPoint: .top,
                               endPoint: .bottom)
                    .ignoresSafeArea() //pour que le grqdient soit vrm partout bleu vers gris

                VStack(spacing: 20) { // entre le rond et le welcome
                    Image(systemName: "building.2.crop.circle") // dans base de donees
                        .resizable() // pour dimensionner apres
                        .scaledToFit() // le ratio de limage
                        .frame(width: 120, height: 120) //grace a resizable et scaled
                        .foregroundColor(.blue)
                        .shadow(radius: 10)

                    VStack(spacing: 4) { // pour aligner les elements les uns en dessous des autres
                        Text("Welcome to")
                            .font(.title2)
                            .foregroundColor(.blue)

                        Text("COREX")
                            .font(.system(size: 40, weight: .bold, design: .rounded)) //pas beau en title
                            .foregroundColor(.blue)
                    }

                    NavigationLink(destination: ChatView()) {
                        Text("Start Chat")
                            .font(.headline)
                            .padding() //agrandit lespace autour
                            .frame(width: 200) // longueur du bleu autour
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)//coins
                            .shadow(radius: 4)
                    }
                }
                .padding()
            }
        }
    }
}
